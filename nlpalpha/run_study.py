"""Run the full study: build features, fit the ladder, test H1-H4, report.

Protocol, fixed before any result was looked at:

  splits      official StockNet dates. Train 2014-01-01 to 2015-08-01,
              validation to 2015-10-01, test to 2016-01-01. Every choice --
              early stopping, the TF-IDF regularization constant, which
              variant to report -- is made on validation. Test is scored once
              per model at the end.
  seeds       every neural result is the mean over several seeds, reported
              with its spread. A single-seed number on 20k noisy samples is
              not a result.
  baselines   always-up, momentum, price-only, VADER, TF-IDF. Text has to beat
              the price block to support H1, and beat TF-IDF to justify the
              transformer.

Usage:
    STOCKNET_ROOT=/path/to/stocknet-dataset \\
      PYTHONPATH=. python -m nlpalpha.run_study --cache-dir /tmp/nlpalpha
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import breakeven_cost_bps, run_backtest
from .baselines import (day_documents, momentum_signal, tfidf_logreg,
                        vader_day_scores)
from .data import (VAL_START, TEST_START, TEST_END, TRAIN_START, build_panel,
                   cross_check_prices, load_prices, load_tweets, load_vix,
                   make_split)
from .evaluate import (classification_report, compare_auc, day_block_bootstrap,
                       permutation_test_auc)
from .features import (ALL_FEATURES, ATTENTION_FEATURES, PRICE_FEATURES,
                       assemble, standardize)
from .text_model import Vocab, count_params
from .train import (build_day_tensors, embed_tweets, predict, pretrain_mlm,
                    roc_auc, train_day_model)

REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "nlpalpha"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def prepare(cache_dir: Path, vix_path: str | None,
            alt_price_csv: str | None, cutoff_hour_utc: int = 20):
    """Assemble the modelling frame, caching the expensive parse."""
    cache = cache_dir / f"feat_cut{cutoff_hour_utc}.pkl"
    tweets_cache = cache_dir / "tweets.pkl"
    integrity = {}

    if cache.exists() and tweets_cache.exists():
        _log(f"loading cached frame {cache.name}")
        return pd.read_pickle(cache), pd.read_pickle(tweets_cache), integrity

    _log("loading prices and tweets")
    prices = load_prices()
    tweets = load_tweets()
    if alt_price_csv:
        integrity = cross_check_prices(prices, alt_price_csv)
        _log(f"price cross-check: median_rel_err="
             f"{integrity.get('median_rel_err'):.2e} over "
             f"{integrity.get('n_overlap')} overlapping rows")

    panel = build_panel(prices, tweets, cutoff_hour_utc=cutoff_hour_utc)
    vix = load_vix(vix_path) if vix_path else None
    feat = assemble(panel, prices, tweets, vix=vix)
    cache_dir.mkdir(parents=True, exist_ok=True)
    feat.to_pickle(cache)
    tweets.to_pickle(tweets_cache)
    return feat, tweets, integrity


def get_encoder(feat, tweets, split, cache_dir: Path, mlm_epochs: int,
                seed: int = 0):
    """MLM-pretrain (or load) the encoder and embed every tweet."""
    emb_path = cache_dir / f"tweet_emb_e{mlm_epochs}_s{seed}.npy"
    meta_path = cache_dir / f"mlm_meta_e{mlm_epochs}_s{seed}.json"
    if emb_path.exists() and meta_path.exists():
        _log(f"loading cached tweet embeddings {emb_path.name}")
        return np.load(emb_path), json.loads(meta_path.read_text())

    train_rows = np.concatenate(feat.loc[split.train, "tweet_idx"].to_numpy())
    train_tokens = tweets.loc[sorted(set(train_rows)), "tokens"].tolist()
    _log(f"building vocab from {len(train_tokens)} TRAIN-window tweets only")
    vocab = Vocab.build(train_tokens, min_freq=5)

    _log(f"MLM pretraining: vocab={len(vocab)} epochs={mlm_epochs}")
    encoder, info = pretrain_mlm(train_tokens, vocab, epochs=mlm_epochs,
                                 seed=seed)
    info["vocab_size"] = len(vocab)
    info["encoder_params"] = count_params(encoder)
    info["n_train_tweets"] = len(train_tokens)

    _log("embedding all tweets with the frozen encoder")
    emb = embed_tweets(encoder, vocab, tweets["tokens"].tolist())
    np.save(emb_path, emb)
    meta_path.write_text(json.dumps(info, indent=2))
    return emb, info


# ---------------------------------------------------------------------------
# Model ladder
# ---------------------------------------------------------------------------

def fit_neural(name, x, mask, feats, y, split, d_model, seeds, use_text):
    """Fit one neural configuration across seeds; return per-seed test probs."""
    tr, va, te = split.train, split.val, split.test
    n_features = feats.shape[1] if feats is not None else 0

    def sub(m):
        return (x[m] if use_text else None,
                mask[m] if use_text else None,
                feats[m] if feats is not None else None,
                y[m])

    probs_test, probs_val, val_aucs = [], [], []
    for seed in seeds:
        model, info = train_day_model(sub(tr), sub(va), d_model=d_model,
                                      n_features=n_features,
                                      use_text=use_text, seed=seed)
        xt, mt, ft, _ = sub(te)
        xv, mv, fv, _ = sub(va)
        probs_test.append(predict(model, xt, mt, ft))
        probs_val.append(predict(model, xv, mv, fv))
        val_aucs.append(info["best_val_auc"])
        _log(f"  {name} seed={seed} val_auc={info['best_val_auc']:.4f} "
             f"(epoch {info['best_epoch']})")
    return {"name": name, "prob_test": np.array(probs_test),
            "prob_val": np.array(probs_val), "val_auc": val_aucs}


def summarize(name, y_test, probs_test, dates_test, ref_probs=None) -> dict:
    """Per-seed test metrics plus the seed-ensemble, and significance."""
    per_seed = [classification_report(y_test, p) for p in probs_test]
    ens = probs_test.mean(axis=0)
    out = {
        "model": name,
        "n_seeds": int(len(probs_test)),
        "accuracy_mean": float(np.mean([m["accuracy"] for m in per_seed])),
        "accuracy_std": float(np.std([m["accuracy"] for m in per_seed])),
        "mcc_mean": float(np.mean([m["mcc"] for m in per_seed])),
        "mcc_std": float(np.std([m["mcc"] for m in per_seed])),
        "auc_mean": float(np.mean([m["auc"] for m in per_seed])),
        "auc_std": float(np.std([m["auc"] for m in per_seed])),
        "ensemble": classification_report(y_test, ens),
    }
    out["permutation"] = permutation_test_auc(y_test, ens, dates_test,
                                              n_perm=500)
    if ref_probs is not None:
        out["vs_reference"] = compare_auc(y_test, ref_probs, ens, dates_test)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="/tmp/nlpalpha")
    ap.add_argument("--vix", default=None)
    ap.add_argument("--alt-prices", default=None)
    ap.add_argument("--mlm-epochs", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--cutoff-hour-utc", type=int, default=20)
    ap.add_argument("--costs-bps", type=float, nargs="+", default=[0, 5, 10, 20])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    t_start = time.time()
    report: dict = {"protocol": {
        "splits": {"train_start": str(TRAIN_START.date()),
                   "val_start": str(VAL_START.date()),
                   "test_start": str(TEST_START.date()),
                   "test_end": str(TEST_END.date())},
        "cutoff_hour_utc": args.cutoff_hour_utc,
        "seeds": args.seeds, "mlm_epochs": args.mlm_epochs,
    }}

    feat, tweets, integrity = prepare(cache_dir, args.vix, args.alt_prices,
                                      args.cutoff_hour_utc)
    split = make_split(feat)
    report["data"] = {
        "n_rows": int(len(feat)), "n_tickers": int(feat["ticker"].nunique()),
        "n_tweets": int(len(tweets)),
        "date_range": [str(feat["date"].min().date()),
                       str(feat["date"].max().date())],
        "split_sizes": split.sizes(),
        "median_tweets_per_day": float(feat["n_tweets"].median()),
        "base_rate_up_test": float(feat.loc[split.test, "up"].mean()),
        "frac_in_deadband": float(feat["in_deadband"].mean()),
        "price_integrity": integrity,
    }
    _log(f"frame: {report['data']['n_rows']} rows, splits {split.sizes()}")

    y = feat["up"].to_numpy()
    y_test = y[split.test]
    dates_test = feat.loc[split.test, "date"].to_numpy()

    # ---- feature blocks, standardized on train moments only ---------------
    vader = vader_day_scores(feat, tweets)
    feat = pd.concat([feat, vader], axis=1)
    blocks = {
        "price": PRICE_FEATURES,
        "attention": ATTENTION_FEATURES,
        "vader": list(vader.columns),
        "price+attention": PRICE_FEATURES + ATTENTION_FEATURES,
    }
    std_blocks = {}
    for bname, cols in blocks.items():
        raw = feat.loc[:, cols].to_numpy(dtype=np.float32)
        tr_s, all_s, _, _ = standardize(raw[split.train], raw)
        std_blocks[bname] = np.nan_to_num(all_s).astype(np.float32)

    # ---- text encoder ------------------------------------------------------
    emb, mlm_info = get_encoder(feat, tweets, split, cache_dir,
                                args.mlm_epochs, seed=args.seeds[0])
    report["mlm"] = mlm_info
    x, mask = build_day_tensors(feat, emb)
    d_model = emb.shape[1]

    # ---- ladder ------------------------------------------------------------
    results = {}

    always_up = np.ones(len(y_test))
    results["always_up"] = {
        "model": "always_up",
        "ensemble": classification_report(y_test, always_up * 0.51),
    }
    mom = momentum_signal(feat)[split.test]
    results["momentum_reversal"] = {
        "model": "momentum_reversal",
        "ensemble": classification_report(
            y_test, (mom - mom.mean()) / (mom.std() + 1e-9) * 0.1 + 0.5),
        "auc": roc_auc(y_test, mom),
    }

    _log("fitting price-only baseline")
    price_fit = fit_neural("price_only", x, mask, std_blocks["price"], y, split,
                           d_model, args.seeds, use_text=False)
    price_ens = price_fit["prob_test"].mean(axis=0)
    results["price_only"] = summarize("price_only", y_test,
                                      price_fit["prob_test"], dates_test)

    _log("fitting attention-only (counts, no words)")
    att_fit = fit_neural("attention_only", x, mask, std_blocks["attention"], y,
                         split, d_model, args.seeds, use_text=False)
    results["attention_only"] = summarize("attention_only", y_test,
                                          att_fit["prob_test"], dates_test,
                                          ref_probs=price_ens)

    _log("fitting VADER lexicon baseline")
    vader_fit = fit_neural("vader_lexicon", x, mask, std_blocks["vader"], y,
                           split, d_model, args.seeds, use_text=False)
    results["vader_lexicon"] = summarize("vader_lexicon", y_test,
                                         vader_fit["prob_test"], dates_test,
                                         ref_probs=price_ens)

    _log("fitting TF-IDF + logistic regression")
    docs = day_documents(feat, tweets)
    tf = tfidf_logreg(docs, y, split.train, split.val, split.test)
    results["tfidf_logreg"] = {
        "model": "tfidf_logreg", "val_auc": tf["val_auc"], "C": tf["C"],
        "n_features": tf["n_features"],
        "ensemble": classification_report(y_test, tf["prob_test"]),
        "permutation": permutation_test_auc(y_test, tf["prob_test"],
                                            dates_test, n_perm=500),
        "vs_reference": compare_auc(y_test, price_ens, tf["prob_test"],
                                    dates_test),
    }

    _log("fitting transformer text-only")
    text_fit = fit_neural("transformer_text", x, mask, None, y, split, d_model,
                          args.seeds, use_text=True)
    results["transformer_text"] = summarize("transformer_text", y_test,
                                            text_fit["prob_test"], dates_test,
                                            ref_probs=price_ens)

    # Placebo: the same architecture on embeddings shuffled across tweets.
    # Marginal distribution, tweet counts and pooling behaviour are identical;
    # only the association between a day and its actual words is destroyed.
    # This is not a formality. Attention-pooling a bag of vectors encodes how
    # many vectors there are, so a "text" model can score above chance purely
    # by reading tweet count. Any claim about *semantics* has to clear this
    # bar, not the 0.50 line.
    _log("fitting placebo (embeddings shuffled across tweets)")
    rng = np.random.default_rng(12345)
    emb_shuffled = emb[rng.permutation(len(emb))]
    x_placebo, mask_placebo = build_day_tensors(feat, emb_shuffled)
    placebo_fit = fit_neural("placebo_shuffled_text", x_placebo, mask_placebo,
                             None, y, split, d_model, args.seeds, use_text=True)
    placebo_ens = placebo_fit["prob_test"].mean(axis=0)
    results["placebo_shuffled_text"] = summarize(
        "placebo_shuffled_text", y_test, placebo_fit["prob_test"], dates_test,
        ref_probs=price_ens)
    results["transformer_text"]["vs_placebo"] = compare_auc(
        y_test, placebo_ens, text_fit["prob_test"].mean(axis=0), dates_test)

    _log("fitting transformer text + price")
    comb_fit = fit_neural("transformer_text_price", x, mask,
                          std_blocks["price+attention"], y, split, d_model,
                          args.seeds, use_text=True)
    comb_ens = comb_fit["prob_test"].mean(axis=0)
    results["transformer_text_price"] = summarize(
        "transformer_text_price", y_test, comb_fit["prob_test"], dates_test,
        ref_probs=price_ens)
    report["models"] = results

    # ---- H2: attention predicts magnitude, semantics do not ---------------
    abs_ret = np.abs(feat["fwd_ret"].to_numpy())
    day = feat["date"].to_numpy()
    big = np.zeros(len(feat), dtype=int)
    for d in np.unique(day):
        m = day == d
        if m.sum() >= 4:
            big[m] = (abs_ret[m] > np.median(abs_ret[m])).astype(int)
    tz = np.nan_to_num(feat["tweet_z"].to_numpy())
    report["H2_attention_vs_magnitude"] = {
        "auc_tweet_z_predicts_big_move_test":
            roc_auc(big[split.test], tz[split.test]),
        "permutation": permutation_test_auc(big[split.test], tz[split.test],
                                            dates_test, n_perm=500),
        "auc_tweet_z_predicts_direction_test": roc_auc(y_test, tz[split.test]),
        "note": ("If abnormal chatter predicts |move| but not sign, the tweet "
                 "stream is an attention/volatility signal, not a directional "
                 "one -- tradeable through options, not through a stock "
                 "long/short."),
    }

    # ---- H1/H3: economics --------------------------------------------------
    bt_frame = feat.loc[split.test, ["date", "ticker", "fwd_ret"]].copy()
    signals = {
        "price_only": price_ens,
        "tfidf": tf["prob_test"],
        "transformer_text": text_fit["prob_test"].mean(axis=0),
        "transformer_text_price": comb_ens,
    }
    econ = {}
    for sname, sig in signals.items():
        bt_frame["signal"] = sig
        per_cost = {}
        for c in args.costs_bps:
            r = run_backtest(bt_frame, cost_bps=c, scheme="rank")
            r.pop("_daily", None)
            per_cost[f"{c:g}bps"] = r
        econ[sname] = {
            "by_cost": per_cost,
            "breakeven_cost_bps": breakeven_cost_bps(bt_frame),
        }
    report["H3_economics"] = econ

    # ---- H4: deadband inflation -------------------------------------------
    dead = feat.loc[split.test, "in_deadband"].to_numpy()
    bt_frame["signal"] = comb_ens
    full = run_backtest(bt_frame, cost_bps=0.0); full.pop("_daily", None)
    kept = run_backtest(bt_frame.loc[~dead], cost_bps=0.0); kept.pop("_daily", None)
    report["H4_deadband"] = {
        "frac_test_in_deadband": float(dead.mean()),
        "auc_full_sample": roc_auc(y_test, comb_ens),
        "auc_outside_deadband": roc_auc(y_test[~dead], comb_ens[~dead]),
        "sharpe_gross_full_sample": full["sharpe_gross"],
        "sharpe_gross_outside_deadband": kept["sharpe_gross"],
        "note": ("Evaluating only outside the deadband removes the small moves "
                 "that a real book still has to trade. Any gap here is the "
                 "amount by which the published protocol flatters economics."),
    }

    report["wall_sec"] = round(time.time() - t_start, 1)
    out_path = Path(args.out) if args.out else REPORT_DIR / "study.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    _log(f"wrote {out_path}  ({report['wall_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
