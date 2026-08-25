# Findings: text does not predict next-day equity direction here

Result of the pre-registered study in `README.md`. Test split scored once:
**2,722 ticker-days, 62 trading days, 2015-10-01 → 2016-01-01**, 85 tickers.
Unconditional up-rate 52.09%. Encoder: 1.85M params, 3 MLM epochs
(perplexity 360, token accuracy 19.5%). Full output: `reports/nlpalpha/study.json`.

**Headline: H1 is rejected. Financial text adds no tradeable next-day
directional information beyond price and volume in this sample, and the
transformer does not beat a placebo that has had its semantics destroyed.**

## The ladder

| model | acc | MCC | AUC | AUC sd | within-day perm p |
| --- | --- | --- | --- | --- | --- |
| always_up | .5209 | .0000 | .5000 | — | — |
| momentum_reversal | .4904 | −.0215 | .4854 | — | — |
| **price_only** | **.5301** | **.0434** | **.5606** | .0020 | 0.966 |
| attention_only | .5202 | .0081 | .4827 | .0043 | 0.918 |
| vader_lexicon | .5209 | .0028 | .4721 | .0044 | 0.986 |
| tfidf_logreg | .4923 | −.0166 | .5001 | — | 0.539 |
| transformer_text | .5209 | .0000 | .4954 | .0038 | 0.697 |
| placebo_shuffled_text | .5209 | .0000 | .5056 | .0137 | 0.359 |
| transformer_text_price | .5338 | .0534 | .5860 | .0038 | 0.938 |

Every text-only arm sits at or below chance. TF-IDF lands on 0.5001 — chance to
four decimals. VADER is *below* chance. `transformer_text` has MCC of exactly
0.0000 because it collapsed to predicting "up" for every row, which given a
52.09% base rate is the loss-minimizing constant.

## The methodological finding, which matters more than the null

**For every single model, the within-day permutation null mean exceeds the
observed AUC.** price_only: observed 0.5654, null 0.5748 ± 0.0052.
transformer_text_price: observed 0.5867, null 0.5931 ± 0.0041.

Shuffling predictions *within each date* destroys stock-selection skill while
leaving every market-wide move intact. That the null scores *higher* means the
pooled AUC of ~0.56–0.59 is entirely a day-level effect — the models know
which **days** the market rose, not which **stocks** outperformed — and their
cross-sectional ranking is, if anything, slightly counterproductive.

This is why AUC 0.586 coexists with a gross Sharpe of −2.5 below. A
dollar-neutral book can only monetize cross-sectional skill, and there is none
here. **Pooled panel AUC, the standard metric in this literature, is not a
measure of stock-selection skill**, and a paper reporting only pooled accuracy
on a stock panel cannot distinguish the two.

## Hypotheses

**H1 — text adds directional information beyond price/volume. REJECTED.**
Day-clustered bootstrap on ΔAUC vs `price_only`: every text arm is
significantly *negative* — transformer_text −0.0771 (95% CI [−0.152, −0.000]),
vader −0.0991, tfidf −0.0652.

The decisive test is the placebo. Real text versus embeddings shuffled across
tweets: **ΔAUC = −0.0293, 95% CI [−0.060, +0.002], p(Δ≤0) = 0.965.** The model
fed genuine semantics does no better — point-estimate worse — than the same
architecture fed the same vectors with the text↔day link destroyed. Whatever
the text arm was doing, it was not reading meaning. Without this control the
placebo's own 0.5176 AUC would have looked like a modest text signal.

The one nominally positive result, `transformer_text_price` at ΔAUC +0.0213
(p(Δ≤0) = 0.029), does **not** rescue H1, for three reasons: it is confounded
(that arm adds the *attention* feature block as well as text, so the delta is
not attributable to text), its own within-day permutation p is 0.938 (the gain
is not cross-sectional), and it loses money in the backtest.

**H2 — attention, not sentiment. SUPPORTED in its magnitude half; the strict
form fails.** Abnormal tweet volume `tweet_z`, 2,000 permutations:

| target | AUC | null mean | p |
| --- | --- | --- | --- |
| big move (\|return\| above that day's median) | .5272 | .5001 | **0.0035** |
| direction | .5394 | .5241 | 0.038 |

Magnitude is genuinely cross-sectional — the null sits at 0.5001, so the whole
+0.027 is stock selection, and p = 0.0035 survives Bonferroni correction for
the ~12 tests reported here (threshold ≈0.004). Direction does **not**: its
null is already 0.5241 (mostly a day-level effect), the excess is +0.015, and
p = 0.038 does not survive correction. So the strict claim "predicts magnitude
but not direction" is too strong — but the *robust* signal is magnitude, and
the directional one is marginal and economically trivial. Semantic features
(VADER, TF-IDF, transformer) predict neither. Chatter is a volatility signal,
which is tradeable through options, not through a stock long/short.

**H3 — costs kill it. SUPPORTED.**

| signal | gross Sharpe | net @5bp | net @10bp | net @20bp | breakeven |
| --- | --- | --- | --- | --- | --- |
| price_only | −3.32 | −6.04 | −8.78 | −14.22 | 0.0bp |
| tfidf | −0.67 | −6.61 | −12.51 | −24.08 | 0.0bp |
| transformer_text | **+1.87** | −2.68 | −7.24 | −16.35 | 2.1bp |
| transformer_text_price | −2.51 | −5.69 | −8.90 | −15.39 | 0.0bp |

The only positive gross Sharpe, transformer_text's +1.87, is not significant
(95% CI [−2.14, +5.64], p(Sharpe≤0) = 0.172) and comes from a model whose AUC
is 0.4954 — below chance. It is 62 days of noise, and a useful reminder that a
short-window Sharpe can look good while the underlying classifier does not
work. Its breakeven of 2.1bp is below the 5–20bp round-trip realistic for US
large caps, so it is untradeable even taken at face value.

**H4 — the deadband inflates. SUPPORTED for metrics, NOT for economics.**
33.6% of test rows fall inside StockNet's ±0.5% band. Dropping them lifts AUC
from 0.5867 to 0.6006 (+0.014) — the published protocol does flatter
classification. But gross Sharpe *falls* from −2.51 to −3.58, so in this sample
the deadband is not hiding economic value. The metric inflation is real; the
economic inflation I predicted is not.

## What would change these conclusions

Stated plainly, because the null is only as strong as its power:

- **62 test days is short.** Sharpe CIs span roughly ±4. The negative Sharpes
  are *not* evidence of anti-alpha; they are consistent with zero. H3's real
  support is the breakeven costs (≈0bp), which are far less noisy.
- **Text is sparse.** The median ticker-day carries 2 tweets. With so little
  text per prediction, a null is unsurprising — this bounds any conclusion to
  "not extractable at this density", not "not present in text generally".
- **The encoder is small and frozen** (1.85M params, MLM token accuracy 19.5%).
  End-to-end fine-tuning or a larger model could do better. Against that: TF-IDF
  reaches exactly chance, and TF-IDF does not have a capacity problem — which
  suggests a low ceiling in this corpus rather than an underfit model.
- **One MLM seed.** Seeds vary the day-level model only; encoder pretraining
  was run once.
- **2014–2016 Twitter cashtags** are one text source in one regime. Filings,
  earnings-call transcripts and newswire have different density and latency.

## What to do next

1. **Retarget to volatility.** H2's magnitude result is the only thing here
   that survives correction. Predicting |move| from abnormal chatter, traded
   through straddles rather than stock, is the hypothesis this data supports —
   and it connects directly to the 0DTE options work in `odte/`.
2. **Report cross-sectional metrics, not pooled AUC**, in anything downstream.
   The permutation gap above shows how badly pooled AUC misleads on a panel.
3. **Keep the placebo arm** in any future text model. It caught what would
   otherwise have read as a real signal.
