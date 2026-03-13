import pandas as pd
import numpy as np
from signals.base import Signal


class Backtest:
    def __init__(
        self,
        signal: Signal,
        initial_cash: float = 10_000,
        allow_short: bool = False,
        fee_bps: float = 20,
        borrow_rate: float = 0.10,
        trading_days: float = 252
    ):
        self.signal = signal
        self.initial_cash = initial_cash
        self.allow_short = allow_short
        self.fee_bps = fee_bps
        self.borrow_rate = borrow_rate
        self.trading_days = trading_days

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.signal.calc(df)

        df["position"] = self._calc_position(df["signal"])

        df["returns"] = df["Close"].pct_change()

        # KEY POINT: assumption is buys/sells take place on t+1 (e.g. day AFTER the signal) using the next close
        # KEY POINT: the sign of returns is flipped if we are short, via the "position" column (+1 or -1)
        df["strategy"] = df["returns"] * df["position"].shift(1)
        
        # apply fee drag on days where position changes (entry or exit)
        fee = self.fee_bps / 10_000
        position_changed = df["position"] != df["position"].shift(1)
        df.loc[position_changed, "strategy"] -= fee

        # daily borrow cost on short days: default 10% p.a. / e.g. 252 trading days
        daily_borrow = self.borrow_rate / self.trading_days
        df["borrow_cost"] = np.where(df["position"].shift(1) == -1, daily_borrow, 0)
        df["strategy"] -= df["borrow_cost"]

        # running total of portfolio equity
        df["equity"] = self.initial_cash * (1 + df["strategy"].fillna(0)).cumprod()

        return df

    def _calc_position(self, signals: pd.Series) -> pd.Series:
        # Convert from array of signals to positions, e.g.
        # From: 0000100000
        # To:   0000111111
        position = signals.replace(0, np.nan).ffill().fillna(0)
        if not self.allow_short:
            position = position.clip(0, 1)
        return position.astype(int)

    def summary(self, df: pd.DataFrame, start: str = None, end: str = None) -> dict:
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]

        total_return = (df["equity"].iloc[-1] / self.initial_cash - 1) * 100

        buy_hold_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100

        mean_daily = df["strategy"].mean()
        std_daily = df["strategy"].std()
        sharpe = (mean_daily / std_daily * np.sqrt(self.trading_days)) if std_daily > 0 else 0.0

        rolling_max = df["equity"].cummax()
        drawdown = (df["equity"] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        n_trades, win_rate = self._trade_stats(df)

        return {
            "total_return":    round(total_return, 2),
            "buy_hold_return": round(buy_hold_return, 2),
            "sharpe_ratio":    round(sharpe, 3),
            "max_drawdown":    round(max_drawdown, 2),
            "n_trades":        n_trades,
            "win_rate":        round(win_rate, 2),
        }

    def _trade_stats(self, df: pd.DataFrame) -> tuple[int, float]:
        trades = []
        entry_price = None
        entry_side = None

        for _, row in df.iterrows():
            if row["signal"] == 1:
                if entry_side == -1:
                    # closing a short: win if price fell
                    trades.append(row["Close"] < entry_price)
                    entry_price = None
                    entry_side = None
                # open long
                entry_price = row["Close"]
                entry_side = 1
            elif row["signal"] == -1:
                if entry_side == 1:
                    # closing a long: win if price rose
                    trades.append(row["Close"] > entry_price)
                    entry_price = None
                    entry_side = None
                if self.allow_short:
                    # open short
                    entry_price = row["Close"]
                    entry_side = -1

        if not trades:
            return 0, 0.0

        return len(trades), sum(trades) / len(trades) * 100
