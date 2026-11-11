"""Bot settings. Defaults are deliberately small and paper-only."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "data" / "live"

# IB Gateway listens on 4002 (paper) / 4001 (live); TWS on 7497 / 7496.
PORTS = {"paper": 4002, "live": 4001}


@dataclass
class Config:
    mode: str = "paper"                 # paper | replay | live (live needs allow_live)
    allow_live: bool = False            # real money: only with the user's explicit go
    symbol: str = "XSP"                 # traded: XSP (1/10 SPX) or SPX
    signal_symbol: str = "SPX"          # quotes the model reads (most liquid)
    host: str = "127.0.0.1"
    port: int | None = None             # default from PORTS[mode]
    client_id: int = 17

    # strategy: #1 sell-only (ridge + calendar + context), the best unseen-data variant
    decision_times: tuple = ("10:00", "11:00", "12:00", "13:00", "14:00", "15:00")
    margin: float = 0.15                # sell only if predicted realized/implied < 1 - margin
    post_event_min: int = 60            # no new short within 60 min after a scheduled event

    # risk
    max_loss_per_trade: float = 300.0   # $ worst case of one trade (whole contracts)
    max_loss_per_day: float = 600.0     # $ worst case of all trades opened today
    max_trades_per_day: int = 2
    max_qty: int = 1                    # one XSP lot per trade until >= 1 year of live data (review)
    pause_drawdown: float = 1500.0      # $ realized drawdown that pauses trading
    pause_sessions: int = 10

    # execution: start at mid, give up at most `max_concession` of the
    # mid-to-touch gap, in `walk_steps` steps `walk_wait_s` seconds apart
    walk_steps: int = 5
    walk_wait_s: int = 30
    max_concession: float = 0.5
    combo_tick: float | None = None     # None: 0.01 for XSP, 0.05 for SPX
    max_attempts_per_day: int = 6       # orders placed (filled or not)

    # quotes: every strike within +/- inner_band, then every outer_step to +/- band
    inner_band: float = 0.012
    band: float = 0.03
    outer_step: float = 25.0
    max_lines: int = 95                 # IBKR default: 100 simultaneous quote lines (SPX + XSP)
    shadow_indexes: tuple = ("NDX", "RUT")   # shadow only, extra quote lines (odte/live/xshadow.py)
    xsp_halfwidth: int = 8              # XSP strikes recorded: nearest-the-money +/- 8 points, every 1

    def spx_lines(self) -> int:
        return self.max_lines - (2 * (2 * self.xsp_halfwidth + 1) if self.symbol == "XSP" else 0)

    kill_switch: Path = field(default_factory=lambda: LIVE / "STOP")
    journal_dir: Path = field(default_factory=lambda: LIVE / "journal")
    chain_dir: Path = field(default_factory=lambda: LIVE / "chains")
    model_path: Path = field(default_factory=lambda: LIVE / "model.pkl")

    def tick(self) -> float:
        return self.combo_tick or (0.01 if self.symbol == "XSP" else 0.05)

    def resolved_port(self) -> int:
        return self.port or PORTS.get(self.mode, 4002)

    def check(self) -> None:
        if self.mode == "live" and not self.allow_live:
            raise SystemExit("REFUSED: live trading needs allow_live=True, set only on the "
                             "user's explicit go.")
        if self.symbol not in ("SPX", "XSP"):
            raise SystemExit(f"unsupported symbol {self.symbol}")
