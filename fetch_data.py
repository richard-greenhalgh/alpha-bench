import os
import pandas as pd
import yfinance as yf


class DataFetcher:
    def __init__(self, tickers_path: str, output_dir: str):
        self.tickers_path = tickers_path
        self.output_dir = output_dir
        self.tickers = self.load_tickers()

    def load_tickers(self) -> pd.DataFrame:
        return pd.read_csv(self.tickers_path)

    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        df.index.name = "Date"
        return df

    def save(self, df: pd.DataFrame, ticker: str) -> None:
        safe_name = ticker.replace("^", "").replace("=", "_")
        path = os.path.join(self.output_dir, f"{safe_name}.csv")
        df.to_csv(path)
        print(f"  Saved {ticker} -> {path}")

    def fetch_all(self, start: str, end: str) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        for _, row in self.tickers.iterrows():
            ticker = row["ticker"]
            print(f"Fetching {row['name']} ({ticker})...")
            df = self.fetch(ticker, start, end)
            self.save(df, ticker)


if __name__ == "__main__":
    fetcher = DataFetcher(
        tickers_path="data/tickers.csv",
        output_dir="data/raw",
    )
    fetcher.fetch_all(start="2020-01-01", end="2024-12-31")
