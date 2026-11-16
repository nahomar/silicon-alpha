from filings import edgar


def test_parse_master():
    text = ("Description: Master Index\n\nCIK|Company Name|Form Type|Date Filed|Filename\n"
            "--------------------------------------------------------------------------------\n"
            "1000045|NICHOLAS FINANCIAL INC|10-K|2020-06-15|edgar/data/1000045/0001564590-20-028981.txt\n"
            "1000097|KINGDON CAPITAL|13F-HR|2020-05-14|edgar/data/1000097/0000919574-20-003740.txt\n")
    d = edgar.parse_master(text)
    assert len(d) == 2 and d.loc[0, "accession"] == "0001564590-20-028981" and d.loc[0, "cik"] == 1000045
