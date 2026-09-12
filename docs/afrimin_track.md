# AFRIMIN — African resource flows → global price transmission

**Status: design + one decisive probe. No tradeable claim is made or implied.**

This is a **separate research track** from the 0DTE engine, not Phase 9 of it.
It shares the repo and the tooling; it shares nothing else. Different horizon
(days–months vs microseconds–hours), different data, different instruments,
different execution. It is labeled as its own track specifically so it does not
become the second unlabeled orthogonal program in this repo — see `STATE.md` on
the pre-0DTE sentiment scraper.

---

## 1. The claim, split three ways

The motivating intuition is *"Africa holds ~30% of world mineral reserves and it
isn't priced right."* The reserve share is defensible and citable (UNEP / African
Development Bank; Africa also holds ~90% of PGM and chromium reserves, and the
DRC alone is ~70% of mined cobalt). **"Not priced right" is not one claim — it is
three**, and they are not equally actionable:

| # | Claim | Type | Tradeable? |
|---|---|---|---|
| **(a)** | African producers capture too little of the value chain — raw ore out, margin captured downstream | political economy | **No.** True and well-documented, but you cannot trade unfairness. |
| **(b)** | African resource assets are systematically undervalued vs fundamentals | asset pricing | **Probably not.** The discount is usually the correct price of expropriation, FX-inconvertibility, governance and illiquidity risk. A cheap asset with real tail risk is not mispriced. |
| **(c)** | African supply events reach global price discovery **with a lag** | market microstructure | **Yes — if it survives testing.** This is the only version this track tests for signal. |

**(a) is the research deliverable. (c) is the signal deliverable.** They are
staged: (a) is buildable today from free annual data; (c) is gated on the probe
below. (b) is explicitly out of scope — testing it well requires a risk model we
do not have, and a naive version would just rediscover the risk premium.

### Why (c) is plausible here specifically

Not because "African markets are inefficient" in the abstract — that is a lazy
premise. Because of a concrete asymmetry: the *events* originate in jurisdictions
thinly covered by Western real-time data infrastructure, while the *prices* form
on the LME, COMEX, LSE and NYSE. Zambian grid rationing, Transnet rail failures,
DRC export bans, Guinean political disruption, artisanal-mining shutdowns —
these are not on any HRT dashboard. That gap is the entire thesis.

This is the same strategic framing `STATE.md` already commits to:
*"uncontested niches > head-to-head."*

---

## 2. The binding constraint: frequency and lag, not volume

`docs/signal_probe_result.md` records why the OPRA probe stalled: yfinance
exposes *current* option-chain snapshots, which have **no time axis to compute
returns from**.

**Most African mineral data has exactly this defect.** USGS Mineral Commodity
Summaries, BGS World Mineral Statistics, EITI reports, UN Comtrade — annual or
monthly, with multi-month publication lag, at country granularity. Excellent for
claim (a). Structurally incapable of supporting claim (c).

Building a 54-country ingestion layer against annual reserve data would produce a
large, clean, well-tested dataset that **cannot answer the question that
motivated it.** This track is designed to make that failure mode impossible
rather than merely discouraged:

> `afrimin/data/sources.py` requires every source to declare `frequency` and
> `publication_lag`, and computes `min_signal_horizon` from them. A source cannot
> be used to build a signal whose horizon is shorter than it can physically
> support. The check is mechanical, not a convention.

---

## 3. The decisive probe (gate before any ingestion build)

Mirroring `infra/modal/dir_baseline.py` — the $0 CPU diagnostic that gated a
$20–50k retrain, and which remains the best-judgment call in this repo — **no
general ingestion layer is built until one cheap case shows transmission lag.**

**Chosen case: Eskom loadshedding → PGM complex.**

It is the best-instrumented instance of the entire thesis:

| property | value | why it matters |
|---|---|---|
| supply share | South Africa ≈ 70%+ of mined platinum, ≈ 35–40% palladium | a real supply shock, not a marginal producer |
| mechanism | deep-level PGM mining is power-intensive (hoisting, ventilation, refrigeration); Stage 6+ forces curtailment | physically causal, not a data-mined correlation |
| event frequency | **daily** stage 0–8 | has a genuine time axis |
| instrument liquidity | PPLT/PALL ETFs, SBSW/IMPUY/ANGPY ADRs, all US-listed | tradeable if signal exists |

**If lag-transmission does not appear here — daily data, dominant global supply
share, liquid instruments, causal mechanism — the thesis is weak everywhere
else.** That is what makes it decisive rather than merely first.

### Design (leakage controls)

Two failure modes would produce fake signal, and both are handled explicitly:

1. **Look-ahead on the event.** Loadshedding stages are announced and revised
   intraday. The feature is lagged a full trading day (`--lag`, default 1); no
   same-day stage information enters a prediction.

2. **Beta masquerading as alpha.** Predicting *raw* miner returns from
   loadshedding mostly rediscovers "miners move with metals." The probe therefore
   tests the **factor-residual**: miner returns are regressed on the underlying
   metal return and a broad-market return, and the probe asks whether
   loadshedding predicts what is *left over*. Raw returns are reported alongside
   purely as a diagnostic contrast.

Train/eval is a **time split, never shuffled**, matching
`odte/eval/signal_probe.py`.

### Falsification criteria — declared before running

| outcome | reading |
|---|---|
| residual effect indistinguishable from 0 (\|t\| < 2) | **no transmission lag.** Thesis (c) unsupported at this horizon. Do not build ingestion. Do not buy vendor data. |
| residual t ≥ 2, correct sign (more loadshedding → PGM miners underperform) | transmission lag plausible. Proceed to a second independent case before generalizing. |
| residual t ≥ 2, **wrong sign** | treat as a red flag for specification error, not a contrarian signal. |
| accuracy > 60% on daily direction | assume leakage and hunt for it. This regime should not be that predictable. |

The middle row is a licence to run *one more probe*, not to build the platform.

---

## 4. Honest constraints

- **Historical loadshedding stage data is the unmet dependency.** There is no
  free official historical API. EskomSePush offers a free-tier token but is
  oriented to current/near-term schedules, not deep history. The loader therefore
  accepts a user-supplied CSV and **refuses to run on absent data rather than
  fabricating it** (`afrimin/data/eskom.py`). No synthetic stage series is
  generated anywhere in this track — given what a corrupted target already cost
  this repo once (`docs/data_integrity_finding.md`), silently plausible fake data
  is the single most expensive thing we could build.
- **Signal presence ≠ tradeable alpha.** Carried over verbatim from
  `signal_probe_result.md`. A daily-horizon effect must survive spread, borrow
  cost and ADR tracking error before it means anything.
- **Most African exchanges are not the venue.** African resource exposure prices
  on the LME/COMEX/LSE/NYSE. The JSE is the only deeply liquid African venue;
  Nairobi, Lagos and Accra carry wide spreads and capital controls, so signal
  found there may be structurally untradeable.
- **Survivorship and revision.** Commodity and ADR histories are revised and
  delisted names vanish from free providers. Free data is sufficient to
  *falsify*; it is not sufficient to size a position.
- **Causality is asserted, not proven.** The power→PGM mechanism is physically
  reasonable, which is why this case was chosen — but the probe measures
  association, and load-shedding co-moves with South African macro generally.

---

## 5. Staging

| stage | scope | gate to proceed | cost |
|---|---|---|---|
| **0** | this doc + source registry + Eskom→PGM probe | — | $0 |
| **1** | claim (a) research view: Africa supply share + concentration (HHI) per commodity from USGS/Comtrade | none — independently useful, annual data is appropriate here | $0 |
| **2** | second independent transmission case (candidate: DRC cobalt export policy → cobalt/battery complex) | stage 0 probe returns t ≥ 2, correct sign | $0 |
| **3** | general country × commodity ingestion layer | two independent cases transmit | $0 |
| **4** | vendor data (AIS vessel tracking, trade-flow feeds) | stage 3 shows persistent, sized edge | $$ |

Vendor spend is deliberately last. Per the operating decision on this track:
free sources first, escalate only on demonstrated signal.

---

## Reproduce

```bash
# research view (claim (a)) — annual data, no signal claim
PYTHONPATH=. python -m afrimin.research.concentration

# the decisive probe (claim (c)) — requires a loadshedding CSV; see --help
PYTHONPATH=. python -m afrimin.probe.eskom_pgm --stages data/eskom_stages.csv
```

Both are network-dependent (free price data) and therefore **not in CI**, matching
the precedent set by `odte/eval/signal_probe.py`.
