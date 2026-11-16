"""Headlines count only after batch + 15 min and only inside each window."""
import numpy as np
import pandas as pd

from odte.data import news_titles as nt


def test_windows_respect_visibility():
    # batch 14:30 UTC = 10:30 ET (EDT) -> visible 10:45 ET
    news = pd.DataFrame({"visible_utc": pd.to_datetime(["2026-03-18 14:45", "2026-03-18 15:20"]),
                         "sent": [0.5, -1.0], "fear": [0.2, 0.9], "greed": [0.1, 0.0], "tone": [1.0, -2.0]})
    rows = pd.DataFrame({"day": pd.Timestamp("2026-03-18"),
                         "t": pd.to_datetime(["2026-03-18 10:30", "2026-03-18 10:45", "2026-03-18 11:30"])})
    f = nt.features_for(rows, news)
    assert f["nw_n60"].tolist() == [0, 1, 2]
    assert np.isnan(f.loc[0, "nw_sent15"]) and f.loc[1, "nw_sent15"] == 0.5
    assert np.isclose(f.loc[2, "nw_sent60"], -0.25)
