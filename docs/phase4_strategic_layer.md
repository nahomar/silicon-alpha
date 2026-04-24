# Phase-4 Strategic Layer — Hierarchical RL + MORL (design spec, no code)

**Status**: design-only. Zero code in the repo corresponds to this doc yet.
This is the deliberate choice — scaffolding RL infrastructure before a trained
forecaster exists would produce stubs that can never be tested. The spec lives
here so future-us can start from a real design, not from scratch.

Reference the Silicon Alpha goal in `memory/project_silicon_alpha_goal.md` for
the north-star context this ladders up to.

## What Phase-4 adds to the stack

After Phase-2 (pretrain) and Phase-3 (persistent-kernel live inference) are
done, Phase-4 wraps the forecaster + executor in a two-level control policy:

- **High-Level Controller (HLC)** — PPO-based strategic agent. Horizon:
  hours to days. Job: decide *what strategies to run and in what size* given
  the regime. Examples:
  - Front-run FTSE/Russell index reconstitution deletions (~5-day horizon
    alpha against passive ETF forced-selling windows).
  - Enable/disable the SPX 0DTE MM engine based on realized-vol regime.
  - Rotate capital between SPX MM and Polymarket arb based on opportunity
    surface.
- **Low-Level Controller (LLC)** — the Phase-3 microstructure engine.
  Horizon: microseconds. Job: minimize slippage on the *trades* HLC
  authorized.

Read: HLC says "deploy $5M into SPX MM from 09:45 ET until the Fed statement,
cap gamma at ±100k." LLC spends that budget microsecond-by-microsecond.

## Multi-Objective Reinforcement Learning (MORL)

Scalar-P/L reward is too crude for live capital. Phase-4 migrates to a
**vector reward**:

```
R = [ pnl_bps,                     # realized P&L in basis points
      -slippage_bps,               # execution cost vs mid
      -max_drawdown_bps,           # worst peak-to-trough in the episode
      -gamma_risk_pen,             # position gamma in units of $/$ move²
      -capital_concentration_pen,  # Herfindahl across venues/strikes
      +uptime_frac ]               # fraction of session engine was live
```

The policy is optimized over the **Pareto front** of this vector, not a
linear combination. A dominant policy beats others on at least one axis
without being worse on any. The scalarization weights are learned, not fixed.

## Regime adaptation via POW-dTS

**POW-dTS** = Policy Weighting via Discounted Thompson Sampling. Adapts the
policy to non-stationary markets by treating "which strategy to run right now"
as a bandit problem with time-discounted evidence. Key regimes to handle:

- **Positive gamma / mean-reverting** — dealers are long-gamma, market reverts
  to strikes. MM spreads widen is safe; momentum strategies lose.
- **Negative gamma / trending** — dealers short gamma, moves amplify. MM
  spreads narrow is risky; momentum or protective-hedging strategies win.
- **VIX spike / term-structure inversion** — vol blowup, size everything
  down, widen quotes, protect gamma.
- **Event-driven** — FOMC, NFP, OPEX, tariff shocks. Different rules.

POW-dTS learns the weighting online as regimes shift; no manual thresholds.

## Dependency chain — why not now

Phase-4 depends on a chain where each link gates the next:

1. **P2 pretrain done.** 524M transformer trained on real OPRA. Currently
   blocked on multi-node compute access.
2. **P3 live inference** . Persistent CUDA kernels (`odte/kernels/*.cu`)
   wired to a live feed and a broker. Currently scaffold only.
3. **Single-objective live P&L tracking.** A real account doing real trades,
   producing the ground-truth reward data. Currently zero live trades.
4. **Reward-vector instrumentation.** Measure slippage vs a reference,
   drawdown, gamma exposure, venue concentration. Needs (3) to have data.
5. **Replay simulator.** An offline environment built from historical OPRA
   + live-trade logs where PPO can bootstrap before touching real capital.
6. **PPO HLC training.** Millions of simulator episodes. A real infra
   investment (probably another multi-node job).
7. **POW-dTS online deployment.** Only once HLC has a trained baseline
   policy worth weighting.

Any step is useless without the one before. Scaffolding (6) before (1) is
how projects drown.

## What would prematurely scaffolding look like (and why we're not)

**Would** include: a `odte/rl/` directory with `ppo_hlc.py`, `morl_env.py`,
`pow_dts.py` stubs; a fake simulator for testing; config fields for
`reward_weights` on a model that hasn't been trained.

**Why it's a trap**: every stub imports modules that don't work (the
forecaster, the broker, the metrics pipeline). Every time someone edits
those modules, the RL stubs have to be updated. The stubs can never be
tested because the environment doesn't exist. Net effect: negative — more
maintenance burden, zero validation possible, new contributors waste days
trying to understand why `ppo_hlc.train()` can't actually be called.

**What instead**: this doc, and nothing else. When the chain reaches (5)-(6)
for real, we start writing code against a real environment.

## Component sketches — for when it's time

These are sketches only. Do NOT translate into code before the dependency
chain is satisfied.

### HLC interface

```text
HLC.observe(regime_embedding, venue_capacities, current_positions)
     -> strategy_weights: vector over {SPX_MM, polymarket_arb, kalshi_arb,
                                       index_delete_fr, off}
HLC.allocate(strategy_weights, total_capital)
     -> {strategy_id: (size_usd, time_window, risk_limits)}
```

HLC decisions are discrete events (per-epoch or regime-change-triggered),
not per-tick. Output flows to LLC which handles per-tick execution.

### LLC interface (= Phase-3 engine)

```text
LLC.quote(strategy_budget, neural_forecast, book_state)
     -> {cancel_orders, new_orders}
```

Already spec'd in the Silicon Alpha goal.

### MORL objective

```text
R_t = [ pnl_bps_t,
        -slippage_bps_t,
        -max_drawdown_bps_t,
        -gamma_risk_pen_t,
        -concentration_pen_t,
        +uptime_frac_t ]
```

Reward is a vector per time step; the Pareto-front optimizer (CAPQL, MORL-PPO,
or POF-PO variants — pick based on what's on the shelf at the time) learns
a policy that's non-dominated.

### POW-dTS regime adapter

```text
given a pool of trained policies {π_mean_revert, π_trend, π_vix_spike, ...}
  and regime observations o_t:
weights_t = ThompsonSample(posterior_reward_per_policy | o_t, γ)
action_t = weighted-vote(weights_t, {π_i(o_t)})
update posterior with discount factor γ after reward observation
```

γ controls how quickly old evidence is discounted — key knob for
non-stationarity.

## What exists today that's related

- `odte/train/` — pretraining code. Not RL.
- `odte/kernels/` — persistent kernel scaffolds. Not connected to a policy.
- `odte_live_paper.py` — older paper trader, unrelated to TradeFM.
- `mm/` — crypto MM, unrelated to HLC/LLC split.

**Concretely: zero code in the repo today implements any part of this doc.**
That's correct for this phase.

## Exit criteria (when to start building)

Begin Phase-4 code only when all of:

1. Single-asset 524M pretrain loss curve exists and shows meaningful learning.
2. Persistent-kernel Phase-3 runtime produces inference in the 4.6-15.8 µs
   budget against a live feed.
3. At least one live trading account is open with the system producing
   trade logs you can replay.
4. Manually-designed single-objective SPX MM strategy is running live and
   producing positive expectancy (proves the base alpha exists before we
   add RL on top).

Without those four, Phase-4 is premature.

## Related files

- [`docs/cross_asset_fusion.md`](cross_asset_fusion.md) — Phase-2.5 scaffold
  (upstream of this spec)
- [`memory/project_silicon_alpha_goal.md`](../memory/project_silicon_alpha_goal.md)
  — north-star goal
- [`infra/gcp/phase2_a3mega.sh`](../infra/gcp/phase2_a3mega.sh) — Phase-2
  compute gate (must clear before Phase-4 is buildable)
