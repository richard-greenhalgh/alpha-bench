import pandas as pd
from signals.base import Signal


class MACrossover(Signal):
    def __init__(self, n_fast: int, n_slow: int):
        # init parent
        super().__init__(name="MA Crossover")
        self.n_fast = n_fast
        self.n_slow = n_slow

    def calc(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["ma_fast"] = df["Close"].rolling(self.n_fast).mean()
        df["ma_slow"] = df["Close"].rolling(self.n_slow).mean()

        df["signal"] = 0
        df.loc[df["ma_fast"] > df["ma_slow"], "signal"] = 1
        df.loc[df["ma_fast"] < df["ma_slow"], "signal"] = -1

        # a crossover is where the signal changes, not just where fast > slow
        # flag when a cross happens, not all rows with fast > slow
        df["signal"] = df["signal"].diff().clip(-1, 1).fillna(0).astype(int)

        return df
