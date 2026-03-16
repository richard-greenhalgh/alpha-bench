import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest.backtest import Backtest
from signals.ma_crossover import MACrossover


TICKERS_PATH = "data/tickers.csv"
RAW_DIR      = "data/raw"


# --- Data loading ---

def load_tickers() -> pd.DataFrame:
    return pd.read_csv(TICKERS_PATH)


def load_ohlcv(ticker: str) -> pd.DataFrame:
    safe_name = ticker.replace("^", "").replace("=", "_")
    path = os.path.join(RAW_DIR, f"{safe_name}.csv")
    return pd.read_csv(path, index_col="Date", parse_dates=True)


# --- Signal registry ---
# To add a new signal: import it and add an entry here

def get_signal_classes() -> dict:
    return {
        "MA Crossover": MACrossover,
    }


# --- UI helpers ---

def render_signal_params(signal_cls) -> dict:
    # Render a Streamlit widget for each param in the signal's schema
    params = {}
    for p in signal_cls.param_schema():
        if p["type"] == "int":
            params[p["name"]] = st.sidebar.slider(
                p["label"], min_value=p["min"], max_value=p["max"], value=p["default"]
            )
        elif p["type"] == "float":
            params[p["name"]] = st.sidebar.slider(
                p["label"], min_value=float(p["min"]), max_value=float(p["max"]),
                value=float(p["default"]), step=p.get("step", 0.01)
            )
    return params


# --- Summary table ---

def build_summary_table(bt: Backtest, df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # one row per year
    for year, group in df.groupby(df.index.year):
        stats = bt.summary(group)
        rows.append({"period": str(year), **stats})

    # total row across full df
    stats = bt.summary(df)
    rows.append({"period": "Total", **stats})

    return pd.DataFrame(rows).set_index("period")


CHART_FONT = dict(size=18)

# --- Chart ---

def build_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # price series (left axis)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        name="Price", line=dict(color="#aaaaaa", width=1),
        yaxis="y1"
    ))

    # buy signals — green dots
    buys = df[df["signal"] == 1]
    fig.add_trace(go.Scatter(
        x=buys.index, y=buys["Close"],
        name="Buy", mode="markers",
        marker=dict(color="lime", size=8, symbol="circle"),
        yaxis="y1"
    ))

    # sell signals — red dots
    sells = df[df["signal"] == -1]
    fig.add_trace(go.Scatter(
        x=sells.index, y=sells["Close"],
        name="Sell", mode="markers",
        marker=dict(color="red", size=8, symbol="circle"),
        yaxis="y1"
    ))

    # equity curve (right axis)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["equity"],
        name="Equity", line=dict(color="royalblue", width=2),
        yaxis="y2"
    ))

    fig.update_layout(
        title=dict(text="Absolute Performance", font=CHART_FONT),
        xaxis=dict(title=dict(text="Date", font=CHART_FONT), tickfont=CHART_FONT),
        yaxis=dict(title=dict(text="Price", font=CHART_FONT), tickfont=CHART_FONT, side="left"),
        yaxis2=dict(title=dict(text="Equity ($)", font=CHART_FONT), tickfont=CHART_FONT, side="right", overlaying="y"),
        legend=dict(orientation="h", y=1.05, font=CHART_FONT),
        hovermode="x unified",
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig


def build_relative_chart(df: pd.DataFrame) -> go.Figure:
    # relative performance = equity / underlying price, rebased to 100 at first data point
    relative = (df["equity"] / df["Close"]) / (df["equity"].iloc[0] / df["Close"].iloc[0]) * 100

    buys  = df[df["signal"] == 1]
    sells = df[df["signal"] == -1]

    fig = go.Figure()

    # price series (left axis)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        name="Price", line=dict(color="#aaaaaa", width=1),
        yaxis="y1"
    ))

    # buy/sell dots on price (left axis)
    fig.add_trace(go.Scatter(
        x=buys.index, y=buys["Close"],
        name="Buy", mode="markers",
        marker=dict(color="lime", size=8, symbol="circle"),
        yaxis="y1"
    ))
    fig.add_trace(go.Scatter(
        x=sells.index, y=sells["Close"],
        name="Sell", mode="markers",
        marker=dict(color="red", size=8, symbol="circle"),
        yaxis="y1"
    ))

    # relative performance (right axis)
    fig.add_trace(go.Scatter(
        x=df.index, y=relative,
        name="Equity / Underlying", line=dict(color="royalblue", width=2),
        yaxis="y2"
    ))

    # reference line at 100 (breakeven vs underlying)
    fig.add_hline(y=100, line=dict(color="white", width=1, dash="dash"), yref="y2")

    fig.update_layout(
        title=dict(text="Relative Performance", font=CHART_FONT),
        xaxis=dict(title=dict(text="Date", font=CHART_FONT), tickfont=CHART_FONT),
        yaxis=dict(title=dict(text="Price", font=CHART_FONT), tickfont=CHART_FONT, side="left"),
        yaxis2=dict(title=dict(text="Equity / Underlying (rebased to 100)", font=CHART_FONT), tickfont=CHART_FONT, side="right", overlaying="y"),
        legend=dict(orientation="h", y=1.05, font=CHART_FONT),
        hovermode="x unified",
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig


# --- Main ---

def main():
    st.set_page_config(page_title="alpha-bench", layout="wide")

    # reduce default Streamlit top padding
    st.markdown("<style>.block-container { padding-top: 1rem; }</style>", unsafe_allow_html=True)

    st.title("alpha-bench")

    tickers_df  = load_tickers()
    signal_classes = get_signal_classes()

    # --- Sidebar ---
    st.sidebar.header("Configuration")

    ticker = st.sidebar.selectbox(
        "Ticker",
        options=tickers_df["ticker"].tolist(),
        format_func=lambda t: f"{tickers_df.loc[tickers_df['ticker'] == t, 'name'].values[0]} ({t})"
    )

    df_full = load_ohlcv(ticker)
    min_date, max_date = df_full.index.min().date(), df_full.index.max().date()

    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    signal_name = st.sidebar.selectbox("Signal", options=list(signal_classes.keys()))
    signal_cls  = signal_classes[signal_name]

    st.sidebar.subheader("Signal parameters")
    params = render_signal_params(signal_cls)

    # --- Backtest settings (main panel, below title) ---
    with st.container(border=True):
        st.subheader("Backtest Options")
        allow_short  = st.toggle("Allow short selling", value=False)
        c1, c2, c3, c4 = st.columns(4)
        initial_cash  = c1.number_input("Initial cash ($)",     value=10_000, step=1_000, min_value=1)
        fee_bps       = c2.number_input("Fee (bps, per side)",  value=20,     step=1,     min_value=0)
        borrow_rate   = c3.number_input("Borrow rate (% p.a.)", value=10.0,   step=0.5,   min_value=0.0, format="%.1f")
        trading_days  = c4.number_input("Trading days p.a.",    value=252,    step=1,     min_value=1)

    # --- Run backtest ---
    start, end = str(date_range[0]), str(date_range[1])
    df = df_full.loc[start:end].copy()

    signal = signal_cls(**params)
    bt     = Backtest(
        signal=signal,
        initial_cash=initial_cash,
        allow_short=allow_short,
        fee_bps=fee_bps,
        borrow_rate=borrow_rate / 100,
        trading_days=trading_days,
    )
    result = bt.run(df)

    # --- Charts ---
    st.plotly_chart(build_chart(result), use_container_width=True)
    st.plotly_chart(build_relative_chart(result), use_container_width=True)

    # --- Summary table ---
    st.subheader("Performance summary")
    summary_df = build_summary_table(bt, result)
    st.dataframe(summary_df, use_container_width=True)


if __name__ == "__main__":
    main()
