"""Free public data for the carry test (docs/carry_plan.md). Run via `python3 -m carry.backtest fetch`.

Everything here is public: no account, no key, no login. One request at a time with a pause.
Anything a server returns is data only. If a source refuses (401/403/451), it is logged and
skipped; there is no workaround.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "carry"
RAW = DATA / "raw"
S3_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
VISION = "https://data.binance.vision/"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
LAST_MONTH = "2026-10"
SPOT_TAIL_DAYS = pd.date_range("2026-11-01", "2026-11-07", freq="D")   # 7-day look-forward for margin risk
IRX_START, IRX_END = "2019-11-01", "2026-11-03"
CDE_URL = "https://api.exchange.fairx.net/rest/funding-rate"
CDE_SYMBOLS = ("BIPZ30", "ETPZ30")
BITNOMIAL_URL = "https://bitnomial.com/exchange/api/v1/funding-rates/"
BITNOMIAL_BASES = ("BTCUC", "ETHUI", "BUC")
KALSHI_URL = "https://external-api.kalshi.com/trade-api/v2/margin/funding_rates/historical"
HEADERS = {"User-Agent": "carry-research/0.1 (python-requests; public data)"}
PAUSE = 0.5

_log: list = []


def _now() -> dict:
    t = datetime.now(timezone.utc)
    return {"utc": t.isoformat(timespec="seconds"), "local": t.astimezone().isoformat(timespec="seconds")}


def _get(url: str, params: dict | None = None, timeout: int = 60) -> requests.Response:
    time.sleep(PAUSE)
    r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    return r


# ---------------------------------------------------------------------------
# Binance public files (data.binance.vision)
# ---------------------------------------------------------------------------

def list_keys(prefix: str) -> list:
    keys, marker = [], ""
    while True:
        r = _get(S3_LIST, {"delimiter": "/", "prefix": prefix, "marker": marker})
        r.raise_for_status()
        ks = re.findall(r"<Key>(.*?)</Key>", r.text)
        keys += ks
        if "<IsTruncated>true</IsTruncated>" not in r.text or not ks:
            return keys
        marker = ks[-1]


def download_checked(key: str) -> Path:
    """Download one zip and verify it against its .CHECKSUM (SHA-256)."""
    dest = RAW / "binance" / key.split("data/", 1)[1]
    dest.parent.mkdir(parents=True, exist_ok=True)
    ck = _get(VISION + key + ".CHECKSUM")
    ck.raise_for_status()
    want = ck.text.split()[0].strip().lower()
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == want:
        return dest
    r = _get(VISION + key)
    r.raise_for_status()
    got = hashlib.sha256(r.content).hexdigest()
    if got != want:
        raise ValueError(f"checksum mismatch for {key}")
    dest.write_bytes(r.content)
    return dest


def _read_zip_csv(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        df = pd.read_csv(io.BytesIO(z.read(name)), header=None, dtype=str)
    first = df.iloc[:, 0].str.strip()
    return df[first.str.fullmatch(r"\d+")].reset_index(drop=True)      # drop a header row if present


def parse_funding(paths: list, symbol: str) -> pd.DataFrame:
    """Rows: time (UTC, rounded to the minute), rate, interval_h."""
    out = []
    for p in paths:
        df = _read_zip_csv(p)
        t = pd.to_datetime(df[0].astype("int64"), unit="ms", utc=True).dt.round("min").dt.tz_localize(None)
        out.append(pd.DataFrame({"time": t, "rate": df[2].astype(float), "interval_h": df[1].astype(float)}))
    f = pd.concat(out).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    f.insert(0, "symbol", symbol)
    return f


def _ts(col: pd.Series) -> pd.Series:
    v = col.astype("int64")
    unit = "us" if v.iloc[0] > 10 ** 14 else "ms"                    # spot files switched to µs in 2025
    return pd.to_datetime(v, unit=unit, utc=True).dt.tz_localize(None)


def parse_klines(paths: list, symbol: str) -> pd.DataFrame:
    out = []
    for p in paths:
        df = _read_zip_csv(p)
        d = pd.DataFrame({"date": _ts(df[0]).dt.normalize(), "open": df[1].astype(float),
                          "high": df[2].astype(float), "low": df[3].astype(float),
                          "close": df[4].astype(float)})
        out.append(d)
    k = pd.concat(out).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    k.insert(0, "symbol", symbol)
    return k


def _months_through(keys: list, last: str) -> list:
    z = [k for k in keys if k.endswith(".zip")]
    return [k for k in z if re.search(r"(\d{4}-\d{2})\.zip$", k).group(1) <= last]


def fetch_binance() -> dict:
    info = {}
    fund, spot = [], []
    for s in SYMBOLS:
        fk = _months_through(list_keys(f"data/futures/um/monthly/fundingRate/{s}/"), LAST_MONTH)
        fpaths = [download_checked(k) for k in fk]
        fund.append(parse_funding(fpaths, s))
        sk = _months_through(list_keys(f"data/spot/monthly/klines/{s}/1d/"), LAST_MONTH)
        sk = [k for k in sk if re.search(r"(\d{4}-\d{2})\.zip$", k).group(1) >= "2019-11"]
        sk += [f"data/spot/daily/klines/{s}/1d/{s}-1d-{d:%Y-%m-%d}.zip" for d in SPOT_TAIL_DAYS]
        spot.append(parse_klines([download_checked(k) for k in sk], s))
        info[s] = {"funding_files": len(fk), "first_funding_file": fk[0].rsplit("/", 1)[1],
                   "last_funding_file": fk[-1].rsplit("/", 1)[1], "spot_files": len(sk)}
    F, K = pd.concat(fund, ignore_index=True), pd.concat(spot, ignore_index=True)
    F.to_parquet(DATA / "binance_funding.parquet")
    K.to_parquet(DATA / "binance_spot_1d.parquet")
    return info


# ---------------------------------------------------------------------------
# T-bills
# ---------------------------------------------------------------------------

def fetch_irx() -> dict:
    import yfinance as yf
    raw = yf.download("^IRX", start=IRX_START, end=IRX_END, auto_adjust=False, progress=False)
    s = raw["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s.dropna().rename("irx")
    s.to_frame().to_parquet(DATA / "irx.parquet")
    return {"days": int(len(s)), "first": f"{s.index[0]:%Y-%m-%d}", "last": f"{s.index[-1]:%Y-%m-%d}"}


# ---------------------------------------------------------------------------
# US venues
# ---------------------------------------------------------------------------

def probe_cde() -> dict:
    """Coinbase Derivatives funding endpoint. Documented as no-auth; if it asks for a key, stop."""
    out = {}
    for sym in CDE_SYMBOLS:
        r = _get(CDE_URL, {"symbol": sym, "trading_session_date": "2025-07-22"})
        out[sym] = {"status": r.status_code, "body": r.text[:200]}
        if r.status_code != 200:
            continue
    out["public"] = all(v["status"] == 200 for k, v in out.items() if k in CDE_SYMBOLS)
    return out


def fetch_bitnomial() -> dict:
    rows, info = [], {}
    for base in BITNOMIAL_BASES:
        cursor, n, status = None, 0, None
        while True:
            p = {"base_symbol": base, "begin_time": "2025-01-01T00:00:00Z", "order": "asc", "limit": 100}
            if cursor:
                p["cursor"] = cursor
            r = _get(BITNOMIAL_URL, p)
            status = r.status_code
            if r.status_code != 200:
                break
            j = r.json()
            data = j.get("data") or []
            for d in data:
                rows.append({"base_symbol": base, "product_id": d.get("product_id"),
                             "interval_start": d.get("interval_start"), "interval_end": d.get("interval_end"),
                             "rate": float(d["funding_rate"])})
            n += len(data)
            cursor = (j.get("pagination") or {}).get("cursor")
            cursor = unquote(cursor) if cursor else None          # served URL-encoded; requests re-encodes
            if not data or not cursor:
                break
        info[base] = {"status": status, "rows": n}
    if rows:
        df = pd.DataFrame(rows)
        for c in ("interval_start", "interval_end"):
            df[c] = pd.to_datetime(df[c], utc=True).dt.round("min").dt.tz_localize(None)
        df = df.drop_duplicates(["base_symbol", "product_id", "interval_end"]).sort_values(["base_symbol", "interval_end"])
        df.to_parquet(DATA / "bitnomial_funding.parquet")
    return info


def fetch_kalshi() -> dict:
    r = _get(KALSHI_URL)
    info = {"status": r.status_code}
    if r.status_code != 200:
        return info
    fr = r.json().get("funding_rates") or []
    info["rows"] = len(fr)
    if fr:
        df = pd.DataFrame(fr)
        df["time"] = pd.to_datetime(df["funding_time"], utc=True).dt.round("min").dt.tz_localize(None)
        df["rate"] = df["funding_rate"].astype(float)
        df = df[["market_ticker", "time", "rate"]].drop_duplicates(["market_ticker", "time"]).sort_values(["market_ticker", "time"])
        df.to_parquet(DATA / "kalshi_funding.parquet")
        info["markets"] = sorted(df["market_ticker"].unique().tolist())
    return info


def fetch_all() -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    log = {"started": _now()}
    log["binance"] = fetch_binance()
    log["binance_fapi_note"] = "fapi.binance.com returns HTTP 451 from this Mac (geo-blocked); public files only."
    log["irx"] = fetch_irx()
    log["coinbase_derivatives"] = probe_cde()
    log["bitnomial"] = fetch_bitnomial()
    log["kalshi"] = fetch_kalshi()
    log["finished"] = _now()
    (DATA / "fetch_log.json").write_text(json.dumps(log, indent=2, default=str))
    return log
