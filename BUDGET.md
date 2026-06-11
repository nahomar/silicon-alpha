# Budget Alpha Engine — cost playbook

A from-scratch 0DTE stack that gives up microsecond latency to save ~$250k.
Every piece the HRT-scale plan called for has a cheaper equivalent wired in.

## One-command run (zero paid subscriptions)

```bash
cd ~/silicon-alpha && source .venv/bin/activate
pip install -r requirements.txt
pip install numba                    # optional; unlocks 50-400× on hot paths
python odte_budget.py --device mps   # or --device cuda on an A100 spot
```

Outputs `reports/odte_budget_<ts>.json` with training loss, DML Greek error,
RL MM PnL, and a sample-book margin requirement under the 2026 SEC rule.

## Budget component map

| HRT-scale plan | Budget equivalent | Cost |
|---|---|---|
| `odte/kernels/persistent_decode.cu` (H100 kernel) | `odte/accel/numba_kernels.py` (Numba JIT) | $0 |
| `odte/kernels/rdma_ingest.cu` (GPUDirect RDMA) | `feeds/fmp.py` + `feeds/polygon_feed.py::options_live_iter` (polling) | $19-$199/mo |
| `odte/train/distributed.py` (8×H100, $80-120k) | `odte/train/pretrain_tradefm.py` on RTX 4090 / A100 spot | $1.50-$3/hr |
| FP8 Transformer Engine | bf16 + optional Liger-Kernel (20% speedup, -60% mem) | free OSS |
| CBOE DataShop (~$10k/mo) | `odte/synth_options.py` + `feeds/fmp.py` | $19-$199/mo |
| 1T-token trillion-token corpus | `odte/data/mixer.py` WeightedShardTokenDataset (90/10 quality weighting) | free |
| PDT $25k wealth barrier | 2026 dynamic intraday margin via `odte/exec/intraday_margin.py` | est. $500-$5k equity |
| 524M TradeFM | `configs/tradefm_budget.yml` (40M params) | trains <48h on 4090 |

## Acceleration cheat-sheet

### CPU hot paths — Numba
```python
from odte.accel import fused_bin_numba, microprice_numba, ofi_numba
```
Measured speedups on M-class Mac (see `python -m odte.accel.bench`):
```
fused_bin   numpy=12.31ms  numba=0.21ms   58×
microprice  numpy= 2.43ms  numba=0.01ms  243×
ofi         numpy=48.71ms  numba=0.31ms  157×
markout     numpy= 8.42ms  numba=0.14ms   60×
```
Real numbers will differ by machine; run `python -m odte.accel.bench` to verify.

### CUDA hot paths — Liger-Kernel (optional)
```bash
pip install liger-kernel    # CUDA only
```
```python
from odte.accel import patch_tradefm_with_liger
model = TradeFM(cfg)
patch_tradefm_with_liger(model)      # replaces RMSNorm + SwiGLU blocks
```
Published gains: +20% train throughput, -60% activation memory.
No effect on Mac; patch is a logged no-op.

## Data tiers — pick one (or combine)

| Tier | Source | Latency | Depth | Cost |
|---|---|---|---|---|
| 0 — synth | `odte.synth_options.generate_session` | instant | full chain | free |
| 1 — FMP Ultimate | `feeds/fmp.FMPFeed` | 1-5s poll | NBBO + Greeks | $19-$29/mo |
| 2 — Polygon Options Advanced | `feeds/polygon_feed.PolygonFeed.options_live_iter` | 15-min on Starter / real-time on Adv | NBBO + snapshot | $199+/mo |
| 3 — Databento OPRA | `feeds/databento_opra.DatabentoOPRAFeed` | real-time MBO | full depth | paid |
| 4 — CBOE DataShop (HRT-scale) | `feeds/cboe_datashop.CBOEDataShopReplay` | historical | L3 | $5-10k/mo |

Every tier speaks the SAME event schema (see `feeds/__init__.py` top-comment),
so you can mix: synth for pretrain, FMP for paper, Polygon for prod paper.

## 2026 dynamic intraday margin

`odte/exec/intraday_margin.py::DynamicIntradayMargin` implements a
piecewise-linear estimator per the SEC's replacement for the $25k PDT
rule. The required live equity for a book is:

```
R = regime_mult · (k_gross·GrossNotional  +  k_delta·|Δ$|  +
                    k_vega·|𝒱|           +  k_gamma·|Γ$²|)
```
Coefficients are configurable so you can mirror your broker's published
intraday schedule exactly. Example:

```python
from odte.exec import DynamicIntradayMargin, OptionPosition
eng = DynamicIntradayMargin()
book = [OptionPosition("SPX260417C05500000", qty=5, spot=5500,
                       delta=0.52, gamma=0.002, vega=0.8)]
state = eng.required(book)
print(state.required_equity)    # ≈ several thousand $, not $25k
```

Always reconcile against broker margin before sizing; this is an estimator.

## Honest trade-offs

- **Latency**: Python + Numba gives you **100 µs – 1 ms** per tick decision.
  That's 100-1000× slower than the persistent CUDA kernel's 15 µs target.
  For mid-frequency 0DTE (signal evolves over seconds-to-minutes), this is
  fine; for latency-arbitrage, it isn't.
- **Data quality**: FMP/Polygon polling loses ≥ 80% of the events a real
  MBO feed would show. Tokenizer quality → model quality → edge. Budget
  tiers mean your alpha has to live in slower-moving features.
- **Model size**: 40M Mini-TradeFM underperforms 524M on rare regimes and
  long-tail microstructure sequences. High-quality mixing (`odte/data/mixer.py`)
  recovers most of the gap — but not all.
- **No CUDA**: the entire `odte/kernels/` tree is optional. Your code
  paths still go through `odte/_kernel.py` and the Numba hot paths,
  which is fine for a research / paper-trading stack. Production HFT
  needs the real kernels.

## Minimum working hardware

| Spend/mo | Dev | Training | Paper-trading |
|---|---|---|---|
| **$0** | Mac (MPS) | Mac — smoke tests only | Mac + Coinbase WS |
| **$150** | Mac | A100 spot, weekends | Mac + FMP $19 |
| **$500** | Mac | A100 spot, 40h/wk | Mac + Polygon Advanced |
| **$2k** | dedicated RTX 4090 box | local 4090 + occasional A100 | FMP + Polygon |
