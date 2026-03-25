from typing import List


def get_universe() -> List[str]:
    """
    Return the stock universe to download.
    Start small for testing, expand later.
    """
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]