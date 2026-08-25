# NLP alpha: does financial text predict next-day equity returns?

A deep-learning NLP study on **real tweets and real prices**, asking whether
text carries next-day directional information beyond price and volume — and
whether any such information survives the cost of trading it.

This file is the **pre-registration**: hypotheses, protocol and decision rules
were written down before the test split was scored. Results live in
`reports/nlpalpha/study.json` and are summarized in `FINDINGS.md`.

## Data

| Source | What | Period |
| --- | --- | --- |
| [StockNet](https://github.com/yumoxu/stocknet-dataset) (Xu & Cohen, ACL 2018) | 106,338 tokenized tweets + Yahoo OHLCV, 88 large-cap US tickers across 9 sectors | 2014-01-01 → 2016-01-01 |
| Kaggle S&P-500 daily | independent price feed, used only to audit the above | 2013-02 → 2018-02 |
| CBOE VIX daily | market-regime control | — |

The corpus ships pre-tokenized and anonymized (handles → `AT_USER`, links →
`URL`). The median ticker-day carries just **2 tweets**; the distribution is
heavily skewed (p90 = 11, max = 555). Text is sparse here, which bounds how
much any model can extract and is part of the finding rather than a nuisance.

**Price audit.** StockNet's closes were checked against the independent feed
across 66,456 overlapping ticker-days: median relative error 2.2e-08. The
~1.4% of days where daily *returns* disagree are concentrated in high-dividend
names (PCG, D, DUK, PPL, PFE, MO) on quarterly dates — ex-dividend days, where
StockNet's `adj_close` is correct and a raw close would print a fake ~1% drop.
The audit is what justifies computing every return from `adj_close`.

## The leakage rule

A trade is placed at the **close of day t** and held to the close of t+1. Its
information set is every tweet stamped at or before **20:00 UTC on day t**.

20:00 UTC is conservative: the US cash close is 21:00 UTC under EST and 20:00
under EDT, so this cutoff is at-or-before the close all year. Tweets after it —
evenings, weekends, holidays — roll forward to the next trading day, which is
correct, because you could not have traded on them sooner.

This is where these studies usually go wrong. Bucketing tweets by calendar day
silently admits every tweet published *after* the close being traded, and no
backtest can detect it — the Sharpe just looks better. `tests/nlpalpha/`
asserts the causal property directly rather than trusting it, and the leaky
calendar-day variant is available (`--cutoff-hour-utc 24`) so the damage can be
measured instead of assumed.

## Model

A hierarchical transformer, trained from scratch in PyTorch:

```
tokens → TweetEncoder (2-layer transformer, [CLS] pooled) → tweet vector
tweet vectors of one ticker-day → attention pooling → day vector
day vector (+ tabular features) → MLP → direction logit
```

Stage 1 is masked language modelling on training-window tweets; stage 2 freezes
the encoder, embeds every tweet once, and trains the day-level pooling and head
on the return task. Freezing costs some accuracy and buys many seeds on CPU,
which matters more — a single-seed deep-learning number on 20k noisy samples is
not evidence.

**Why from scratch rather than fine-tuning FinBERT.** Pretrained financial LMs
were trained on corpora overlapping 2014–2016, so fine-tuning one here imports
look-ahead contamination that no backtest can detect. Training the encoder on
train-window text only makes the causality auditable. (Off-the-shelf weights
are also unreachable from this environment, but that is not the reason.)

Scale honesty: ~1.9M parameters over ~10^5 short texts. A language model
architecturally, not a large one.

## Hypotheses

| | Claim | How it is judged |
| --- | --- | --- |
| **H1** | Tweet semantics carry next-day directional information incremental to price/volume | day-clustered bootstrap CI on ΔAUC vs the price-only model must exclude 0 |
| **H2** | The real signal is *attention*, not *sentiment*: abnormal tweet volume predicts \|move\| but not sign | AUC of `tweet_z` for a big-move label vs for direction |
| **H3** | Any directional edge dies under realistic costs | breakeven cost in bp vs the 5–20bp round-trip realistic for US large caps |
| **H4** | The ±0.5% deadband inflates apparent economic value | gross Sharpe on the full sample vs outside the deadband |

## Protocol

- **Splits** are StockNet's official dates: train 2014-01-01→2015-08-01,
  validation →2015-10-01, test →2016-01-01. Never shuffled across boundaries.
- **Every choice** — early stopping, TF-IDF regularization, which variant to
  report — is made on validation. **Test is scored once per model.**
- **Scaler moments and vocabulary come from train only.** Fitting either on the
  full panel leaks the future.
- **Seeds**: neural results are mean ± spread over 3 seeds.
- **Significance** respects the panel's dependence. Daily stock returns are
  strongly cross-sectionally correlated, so treating 2,722 ticker-days as
  independent would badly overstate significance. Permutation tests shuffle
  *within date*; bootstraps resample *whole days*.

## The ladder

A claim about deep NLP has to outrank every cheaper explanation:

1. `always_up` — the unconditional up-rate is above 50%, so accuracy alone always flatters
2. `momentum_reversal` — yesterday's return
3. `price_only` — the technical block; **H1 requires beating this**
4. `attention_only` — tweet counts, no words read
5. `vader_lexicon` — a fixed sentiment dictionary, no fitting
6. `tfidf_logreg` — bag-of-words + logistic regression; **the transformer must beat this to justify itself**
7. `placebo_shuffled_text` — embeddings shuffled across tweets: identical distribution and counts, no real text↔day link
8. `transformer_text`, `transformer_text_price`

**The placebo is not a formality.** Attention-pooling a bag of vectors encodes
*how many* vectors there are, so a "text" model can score above chance by
reading tweet count alone — in a dry run with entirely random embeddings, the
text arm still reached 0.53 validation AUC. Any claim about semantics must
clear the placebo, not the 0.50 line.

## Reproducing

```bash
git clone https://github.com/yumoxu/stocknet-dataset.git
STOCKNET_ROOT=$PWD/stocknet-dataset PYTHONPATH=. \
  python -m nlpalpha.run_study --cache-dir /tmp/nlpalpha \
    --vix vix-daily.csv --alt-prices all_stocks_5yr.csv \
    --mlm-epochs 3 --seeds 0 1 2
PYTHONPATH=. pytest tests/nlpalpha/ -q     # needs no corpus
```

Requires `torch`, `pandas`, `numpy`, `scikit-learn`, `vaderSentiment`.
