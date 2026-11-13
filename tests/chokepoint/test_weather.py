import pandas as pd

from chokepoint.data import weather


class _Resp:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"header": {"fill_value": -999.0},
                "properties": {"parameter": {
                    "PRECTOTCORR": {"20261101": 3.5, "20261102": -999.0},
                    "T2M_MAX": {"20261101": 30.1, "20261102": 29.0}}}}


class _Session:
    def __init__(self):
        self.params = None

    def get(self, url, params, timeout):
        self.params = params
        return _Resp()


def test_fetch_drops_fill_values_and_names_series():
    s = _Session()
    x = weather.fetch("cod_kolwezi", "2026-11-01", "2026-11-02", s)
    assert s.params["latitude"] == weather.SITES["cod_kolwezi"][0]
    assert s.params["start"] == "20261101" and s.params["end"] == "20261102"
    assert len(x) == 3                                   # the -999 rain day is dropped
    assert set(x["series"]) == {"cod_kolwezi:PRECTOTCORR", "cod_kolwezi:T2M_MAX"}
    assert x["period"].dtype.kind == "M"
    assert x.loc[x["series"] == "cod_kolwezi:PRECTOTCORR", "value"].tolist() == [3.5]


def test_sites_are_valid_coordinates():
    assert len(weather.SITES) >= 50
    for lat, lon, why in weather.SITES.values():
        assert -35 <= lat <= 38 and -26 <= lon <= 52 and why
