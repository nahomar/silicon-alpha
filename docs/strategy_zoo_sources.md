# Strategy zoo: sources and families (compiled 2026-11-10)

Companion to `strategy_zoo_catalog.json`, which holds 72 entries:

| Fidelity | Entries |
|---|---|
| exact | 6 |
| approximate | 52 |
| untestable | 14 |

`odte.research.zoo.load_rules` loads the **58 testable entries**. Each one also builds bounded legs on a synthetic chain. All research was read-only. Nothing was logged into or paid for, and no block was worked around.

## How rules were mapped to the engine (read before trusting a result)

### Strike distances
- **Sigma conversions.** Strikes given in delta use the brief's table (for example, 10Δ is 1.28σ). Strikes given as % OTM or by moneyness were converted using a typical sigma-to-close, as a % of spot:

  | Time (ET) | Sigma-to-close |
  |---|---|
  | 10:00 | 0.72% |
  | 11:00 | 0.66% |
  | 12:00 | 0.59% |
  | 13:00 | 0.51% |
  | 14:00 | 0.42% |
  | 15:00 | 0.29% |
  | 15:30 | 0.21% |

  These assume VIX around 16-18; FlashAlpha measured a median of 0.72% at 10:00. Strikes chosen by credit target (MEIC, BEIC, Cboe's "dollar rule") were converted to about 1.1-1.2σ.
- **Naked positions** (short straddles and strangles, short calls, single long options) are capped with far wings, because the engine refuses unbounded upside. That means 3σ wings for short premium and a 1σ-wide debit spread for single long options.

### Engine conventions
- **Debit spreads.** `short_k` is the distance of the *bought* strike, and `wing` is the width beyond it. This is how `legs_for` builds them.
- **Direction.** It is only set where it changes the side. "with_morning_move" is bullish after an up-move: it sells puts on credit spreads and buys calls on debit spreads.
- **Stops act on the whole position, not per side.** MEIC and BEIC stop each spread independently, and that side-level stop roughly breaks the condor even. It is encoded as a combined `stop_loss: 0.2` (plausible range 0.1-0.25). The brief's literal `1.0` is kept as one variant, `meic_chambless_5x30_sl1`. On a whole condor, 1.0 means exiting at 2x total credit, which is far looser than MEIC's stop.
- **Other exit units.** Elsewhere, "stop at 2x credit" / "200% stop" means a loss of 2x credit (`stop_loss: 2.0`). For long premium, `take_profit` is the gain on the debit (0.75 = +75%).
- **Optional fields.** `exit: {"hold_to_expiry": true}` is only a marker. Conditions the vocabulary cannot express are listed in an extra top-level field, `unexpressed_conditions`. That field sits outside `rule` and the engine ignores it.

### Chinese sources
- 末日轮 is the *monthly* 50ETF expiry, so source samples are about 12 events a year.
- Beijing times were mapped by fraction of the 240-minute session: 10:00 → 10:20, 11:00 → 11:55, 14:30 → 15:10, 14:50 → 15:45 ET.
- One step out of the money (OTM-1) at the expiry-day open was taken as about 1.5σ, and two steps (OTM-2) as about 2.5σ.

## Families

| Family | Testable / untestable | Idea | Main sources |
|---|---|---|---|
| breakeven_multi_entry_ic | 4 / 0 | Several 0DTE condors a day, each side stopped near breakeven | Tammy Chambless MEIC slides (exact times and stops); Sandvand BEIC (Theta Profits); Option Alpha BEIC template |
| atm_iron_fly_practitioner | 9 / 0 | ATM iron fly, mostly in the morning, with small targets or stops | Option Alpha 230k-trade study; Options Cafe live log; Theta Profits "Doc"; tastylive (search summaries); spy-0dte-research |
| otm_iron_condor | 11 / 0 | Fixed-time OTM condor | OptionsPilot (exact, vendor); ORATS example (QQQ); Option Alpha 25k study and Oracle examples (Kirk Du Plessis); Cboe Henry Schwartz; Chinese far-OTM sellers |
| directional_credit_spread | 8 / 0 | One-sided credit spread chosen by direction | Option Alpha opening-range breakout (15/30/60 min) and SMA5 rules; VIX1D GitHub bot; OptionsPilot 13:00 put spread; 金衍君 5Δ put seller |
| short_call_overlay | 2 / 1 | Option leg of the daily 0DTE covered call | OptionMetrics blog (25Δ and %-OTM; skew-timed version untestable) |
| intraday_vrp_short_straddle | 8 / 0 | Short ATM straddle by entry time and regime | Dim-Eraker-Vilkov Table 2 (long-straddle returns by bar, 2012-2023); Almeida-Freire-Hizmeri (low-vol states); Londono-Samadi (FOMC premium) |
| vilkov_moneyness_grid | 6 / 2 | Vilkov "0DTE Trading Rules" structures at fixed moneyness | Vilkov GitHub annotated paper; FirmTape; CXO |
| intraday_momentum_close | 2 / 0 | Last-30-minute momentum from dealer gamma hedging | Gao-Han-Li-Zhou (JFE 2018); Baltussen-Da-Lammers-Martens (JFE 2021) |
| late_day_butterfly | 2 / 0 | ATM butterfly into the close (pin / "lotto fly") | Golez-Jackwerth pinning, with Elms 2026 against; Theta Profits butterfly |
| long_premium_lottery | 3 / 0 | Buy-and-hold long premium (expected negative controls) | Beckmeyer-Branger-Gayda; tastylive last-hour guide; SSE 末日彩票 piece |
| long_premium_early_exit | 3 / 0 | Expiry-day long OTM with early time exits | 余力 50ETF minute study; 期权酱 double-buy |
| dealer_positioning_oi | 0 / 4 | Gamma exposure, walls, max pain, structural levels | SpotGamma; 0-dte.com; Option Alpha GEX trade; ETF期权通 max pain |
| overnight_1dte | 0 / 5 | 1DTE or overnight holds | jwplatta 1DTE IC; OptionMetrics Monday put-write; Muravyev-Ni; 广发 and 一德 T-1 trades |
| discretionary_or_vague | 0 / 2 | Active or unspecified strikes | Roundhill XDTE; Xueqiu 捡烟蒂 (blocked) |

The family counts include deliberate controls that are expected to fail: the long-premium families, `dev_long_straddle_1030` and `oa_ic_first_2h`.

## Source quality notes
- **Self-reported or vendor results, unaudited:**
  - OptionsPilot, ORATS, Option Alpha, Options Cafe, Theta Profits, Sandvand and MEIC (the author's own backtest and live record).
  - spy-0dte-research: one rule chosen from about 450 configurations.
- **Rules read only from a search-engine summary,** because the page itself was unreadable:
  - the tastylive Market Measures entries;
  - the Option Alpha BEIC bot;
  - the Option Alpha public 60-min opening-range-breakout bot filters.
  - These are marked in each entry's notes.
- **Conflicting evidence, worth testing as a pair:**
  - FOMC days: Londono-Samadi find a larger premium on FOMC-spanning options, while OptionsPilot found condors did better skipping FOMC.
  - Entry time: the Option Alpha 25k study favours late entries, while tastylive favours the first hour.
  - Pinning: Golez-Jackwerth found it; Elms 2026 says it is gone.
- **Chinese evidence:** consistent across sources. Long OTM held to the expiry close almost always loses, and far-OTM selling wins 88-100% of the time. But the samples are monthly 50ETF expiries, mostly 2015-2021.

## Sources used (all read directly unless noted)

### Practitioner
- Tammy Chambless MEIC slides, Jul 2023 (S3 PDF, text decoded locally).
- Theta Profits: Sandvand's BEIC, the Doc iron fly and a butterfly article.
- Options Cafe iron-fly results.
- Option Alpha: the 0dte-options-strategy-performance study, the 25k-trade study, the opening-range-breakout article, the "top 0DTE strategies" page, the time-decay study and "How I day trade".
- Cboe Insights: Henry Schwartz's condor.
- OptionsPilot.
- ORATS.
- GitHub `thunderscarf/SPX_0DTE_Options_Selling_Public`.
- OptionMetrics blog (three posts).
- FlashAlpha straddle study (context only).
- tastylive last-hour guide.
- spy-0dte-research.pages.dev.
- 0-dte.com guide and broken-wing transcript (discretionary).
- Roundhill blog and prospectus (search result).
- jwplatta 1DTE post.

### Academic
- Dim-Eraker-Vilkov PDF (WFA portal; Table 2 decoded locally).
- Vilkov `0dte-strategies` GitHub annotated paper.
- FirmTape literature sweep: used for the Almeida-Freire-Hizmeri and Muravyev-Ni numbers.
- Federal Reserve IFDP page (Londono-Samadi, search result).
- EconPapers and EUR pages for Gao et al., Baltussen et al. and Golez-Jackwerth (search results).
- Harbourfront Quant summary of Elms 2026.
- Beckmeyer-Branger-Gayda: SSRN 4404704 abstract (via search) and BNN Bloomberg coverage.
- Bandi-Fusari-Renò: read, but it is a pricing paper with no trading rule.

### Chinese (via a sub-agent; Sina, Eastmoney, Sohu and NetEase reprints)
- 余力 / 力的期权工作室 (minute-level exit study).
- 广发期货 in 期货日报 (strangles and straddles, T-1 to T-10).
- 一德期货 in 期权十 (T-1 straddle).
- SSE investor-education piece in 上海证券报 (末日彩票).
- 金衍君 (lottery buying and 5Δ put selling).
- 期权酱 (85% range seller; double-buy).
- ETF期权通 (max pain).
- 光大证券 金工 "期权卖方策略初探": search snippets only.

## Blocked or unusable (skipped, not worked around)
- **Reddit (all subreddits):** reddit.com refuses our fetcher, and the search tool rejects the domain. r/thetagang, r/options, r/algotrading, r/wallstreetbets and the 0DTE subreddits were **not covered**.
- **SSRN abstract pages:** HTTP 403. Copies were found elsewhere for Vilkov, Dim-Eraker-Vilkov and Almeida et al.
- **medium.com summary of Beckmeyer et al.:** 403.
- **studylib copy of Vilkov:** 429.
- **tastylive episode pages and recapio digests:** they render only a landing page or a loading placeholder.
- **Option Alpha bot templates** (`optionalpha.com/bots/*`): they redirect into the logged-in app.
- **Luckbox 0DTE backtest article:** redirects to a tastylive index page.
- **Unreachable sites:** weeklypaycheck.trade and c.0-dte.com (DNS failure); pulse.bot (connection closed).
- **PDFs whose text could not be extracted:** Almeida et al. (CID fonts), the Numerix 0DTE white paper, the 光大 and 兴证期货 reports.
- **Paywalled:** CXO Advisory summaries; Aeromir and Option Omega MEIC courses (not attempted).
- **Chinese sites:**
  - Zhihu: HTTP 403.
  - Xueqiu: WAF challenge.
  - Douban: security check.
  - futuresonline.com.cn: TLS error.
  - CSDN: 521.
  - stcn: timeout.
- **No usable expiry-day rules found** at the major brokers (华泰, 国泰君安, 中信, 招商, 海通), or for 300ETF, 500ETF or IO options.

## Vocabulary gaps (conditions dropped or proxied; see `unexpressed_conditions`)
1. Strike by **target credit** per side (MEIC, BEIC, Cboe dollar rule, Options Cafe minimum credit).
2. **Per-side (per-spread) stops** and stops on the short leg only; stop tightening through the day.
3. Strike by **% OTM or moneyness**, and wings by delta (Vilkov, OptionMetrics, Option Alpha Oracle, ORATS).
4. **Asymmetric condors**, with different put and call distances or wings (several spy-0dte-research rules), and two-width broken-wing butterflies.
5. **Single legs and ratios:** long call or put, naked short put, ratio spreads, risk reversals (Vilkov's best structures; Chinese lottery buys).
6. **Opening-range triggers:** entry on a breakout any time within a window, and a strike at the range boundary.
7. **Signed move and gap filters** (down days only, gap down) and a day-change-since-prior-close filter.
8. **Multi-day features:** SMA or RSI trend, prior-day move or range ("skip after a 1% day"), realized-vol state, monthly-OpEx flag.
9. **Event-type selection** (FOMC only versus CPI/NFP), plus GDP and quad-witching days.
10. **Implied-surface filters:** 0DTE 25Δ skew, VIX1D level or term structure, implied versus realized volatility.
11. **Underlying-price stops** (spot touches a short or long strike), laddered take-profits and trailing stops.
12. **Overnight / 1DTE entries** from the prior close.
13. **Open interest or dealer gamma** (max pain, GEX walls, negative-gamma days): data we do not have.
