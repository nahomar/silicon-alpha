import numpy as np

from filings import text as tx


def test_minhash_estimates_jaccard():
    a = [f"word{i}x" for i in range(1000)]
    b = a[:800] + [f"other{i}y" for i in range(200)]          # true Jaccard = 800 / 1200
    est = tx.jaccard(tx.minhash(tx.html_to_words(" ".join(a).replace("0", "zero").replace("1", "one"))),
                     tx.minhash(tx.html_to_words(" ".join(b).replace("0", "zero").replace("1", "one"))))
    assert abs(est - 800 / 1200) < 0.12
    same = tx.minhash(["alpha", "beta", "gamma"])
    assert tx.jaccard(same, tx.minhash(["gamma", "beta", "alpha", "alpha"])) == 1.0


def test_cover_page_ticker_and_html_stripping():
    html = ('<html><body><ix:nonNumeric name="dei:TradingSymbol" contextRef="c1">aapl</ix:nonNumeric>'
            '<ix:nonNumeric contextRef="c1" name="dei:SecurityExchangeName">NASDAQ</ix:nonNumeric>'
            '<script>var x = 1;</script><p>Risk factors&nbsp;include competition</p></body></html>')
    f = tx.features(html)
    assert f["ticker"] == "AAPL" and f["exchange"] == "NASDAQ"
    assert tx.html_to_words(html)[-3:] == ["risk", "factors", "include", "competition"][-3:]
