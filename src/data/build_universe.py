from typing import List


def get_universe() -> List[str]:
    """
    Return the stock universe to download.
    Start small for testing, expand later.
    """
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "TSLA", "BRK-B", "UNH", "JPM",
        "V", "PG", "XOM", "HD", "MA",
        "CVX", "ABBV", "PEP", "KO", "AVGO",
        "COST", "MRK", "LLY", "WMT", "BAC",
        "TMO", "MCD", "DIS", "CSCO", "ACN",
        "DHR", "VZ", "CRM", "NKE", "TXN",
        "LIN", "ABT", "CMCSA", "AMD", "QCOM",
        "PM", "ORCL", "MDT", "HON", "UPS",
        "IBM", "CAT", "AMAT", "GS", "BA",
        "SPGI", "ADBE", "INTU", "AMGN", "PLD",
        "LOW", "SBUX", "BLK", "SYK", "ELV",
        "DE", "ISRG", "MDLZ", "GILD", "ADI",
        "LRCX", "TJX", "CB", "MMC", "CI",
        "VRTX", "REGN", "MO", "PGR", "C",
        "ZTS", "DUK", "SO", "BDX", "TGT",
        "BKNG", "USB", "PNC", "CSX", "AON",
        "CL", "SHW", "WM", "EOG", "ETN",
        "ITW", "APD", "MCO", "EMR", "ROP",
        "FDX", "GM", "F", "MS", "AXP"
    ]