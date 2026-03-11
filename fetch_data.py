import os
import pandas as pd
import yfinance as yf


# DataFetcher() functionality to read list of tickers and fetch daily OHLCV data
#   (Open High Low Close Volume) using yfinance
#   tickers follow Yahoo Finance conventions: ^ prefix for indices (^GSPC),
#   =F suffix for futures (GC=F), -USD suffix for crypto (BTC-USD)
#   search for any ticker at https://finance.yahoo.com
class DataFetcher:
    def __init__(self, tickers_path: str, output_dir: str):
        self.tickers_path = tickers_path
        self.output_dir = output_dir
        self.tickers = self.load_tickers()

    # read in list of tickers
    def load_tickers(self) -> pd.DataFrame:
        return pd.read_csv(self.tickers_path)

    # for a single ticker, fetch OHLCV data as df
    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True) # auto_adjust: corrects historical prices for splits/dividends so the series is consistent over time
        df.index.name = "Date"
        return df

    # for a single ticker, save OHLCV data as CSV
    def save(self, df: pd.DataFrame, ticker: str) -> None:
        # clean up raw name for creating the CSV
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

# pull data from data/tickers.csv if run standalone
if __name__ == "__main__":
    fetcher = DataFetcher(
        tickers_path="data/tickers.csv",
        output_dir="data/raw",
    )
    fetcher.fetch_all(start="2020-01-01", end="2024-12-31")
