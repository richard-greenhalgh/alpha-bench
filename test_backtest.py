import os
import pandas as pd
from fetch_data import DataFetcher
from signals.ma_crossover import MACrossover
from backtest.backtest import Backtest


TICKERS_PATH = "data/tickers.csv"
RAW_DIR      = "data/raw"
RESULTS_DIR  = "data/results"


def load_data(ticker_file: str, raw_dir: str) -> dict:
    tickers = pd.read_csv(ticker_file)
    data = {}
    for _, row in tickers.iterrows():
        safe_name = row["ticker"].replace("^", "").replace("=", "_")
        path = os.path.join(raw_dir, f"{safe_name}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
            data[row["ticker"]] = df
        else:
            print(f"Warning: no data file found for {row['ticker']} at {path}")
    return data


def run_single(df: pd.DataFrame, signal: MACrossover, **backtest_kwargs) -> pd.DataFrame:
    bt = Backtest(signal=signal, **backtest_kwargs)
    return bt.run(df)


def save_results(df: pd.DataFrame, label: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{label}.csv")
    df.to_csv(path)
    print(f"  Saved -> {path}")


def print_summary(label: str, summary: dict) -> None:
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    for k, v in summary.items():
        print(f"  {k:<20} {v}")


def main():
    # fetch data if raw dir is empty
    if not any(f.endswith(".csv") for f in os.listdir(RAW_DIR)):
        print("No raw data found, fetching...")
        fetcher = DataFetcher(tickers_path=TICKERS_PATH, output_dir=RAW_DIR)
        fetcher.fetch_all(start="2020-01-01", end="2024-12-31")

    data = load_data(TICKERS_PATH, RAW_DIR)
    ticker = "BTC-USD"
    df = data[ticker]

    signal = MACrossover(n_fast=20, n_slow=50)

    variants = [
        {
            "label":       "BTC-USD_long_only_default_fees",
            "allow_short": False,
            "fee_bps":     20,
            "borrow_rate": 0.10,
            "save":        True,
        },
        {
            "label":       "BTC-USD_long_only_high_fees",
            "allow_short": False,
            "fee_bps":     50,
            "borrow_rate": 0.10,
        },
        {
            "label":       "BTC-USD_long_short_default_fees",
            "allow_short": True,
            "fee_bps":     20,
            "borrow_rate": 0.10,
        },
        {
            "label":       "BTC-USD_long_short_high_fees",
            "allow_short": True,
            "fee_bps":     50,
            "borrow_rate": 0.10,
        },
    ]

    summaries = []

    for v in variants:
        label  = v["label"]
        kwargs = {k: v[k] for k in ("allow_short", "fee_bps", "borrow_rate")}

        bt     = Backtest(signal=signal, **kwargs)
        result = bt.run(df.copy())
        stats  = bt.summary(result)

        print_summary(label, stats)
        save_results(result, label, RESULTS_DIR)
        summaries.append({"label": label, **stats})

    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"\nSummary saved -> {summary_path}")


if __name__ == "__main__":
    main()
