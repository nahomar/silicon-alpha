import pandas as pd

from odte.data import context


class _R:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def test_cboe_reads_last_column_as_close(monkeypatch):
    import requests
    vix = "DATE,OPEN,HIGH,LOW,CLOSE\n11/12/2026,18.0,19.0,17.5,18.70\n11/13/2026,18.78,20.92,17.64,19.30\n"
    vvix = "DATE,VVIX\n11/12/2026,96.71\n11/13/2026,101.53\n"
    monkeypatch.setattr(requests, "get", lambda url, **k: _R(vix if "/VIX_" in url else vvix))
    s = context.cboe("vix", "2026-11-13", "2026-11-14")
    assert list(s.index) == [pd.Timestamp("2026-11-13")] and s.iloc[0] == 19.30
    assert context.cboe("vvix", "2026-11-01", "2026-11-14").iloc[-1] == 101.53
