import yfinance as yf
import streamlit as st
import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page to wide mode so the charts have more horizontal space
st.set_page_config(layout="wide")

# Data Functions
@st.cache_data
def get_sp500_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = df[0]
    tickers = df["Symbol"].to_list()
    tickers_companies_dict = dict(zip(df["Symbol"], df["Security"]))
    return tickers, tickers_companies_dict

@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start, end)
    # Flatten columns if it’s a multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode("utf-8")

# Sidebar Inputs
st.sidebar.header("Stock Parameters")

available_tickers, tickers_companies_dict = get_sp500_components()

ticker = st.sidebar.selectbox(
    "Ticker", available_tickers, format_func=tickers_companies_dict.get
)

start_date = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date.today())

if start_date > end_date:
    st.sidebar.error("The end date must be after the start date.")

# Technical Analysis Inputs
st.sidebar.header("Technical Analysis Parameters")

volume_flag = st.sidebar.checkbox(label="Add Volume")

exp_sma = st.sidebar.expander("SMA")
sma_flag = exp_sma.checkbox(label="Add SMA")
sma_periods = exp_sma.number_input("SMA Periods", min_value=1, max_value=50, value=20)

exp_bb = st.sidebar.expander("Bollinger Bands")
bb_flag = exp_bb.checkbox(label="Add Bollinger Bands")
bb_periods = exp_bb.number_input("BB Periods", min_value=1, max_value=50, value=20)
bb_std = exp_bb.number_input("# of standard deviations", min_value=1, max_value=4, value=2)

exp_rsi = st.sidebar.expander("Relative Strength Index")
rsi_flag = exp_rsi.checkbox(label="Add RSI")
rsi_periods = exp_rsi.number_input("RSI Periods", min_value=1, max_value=50, value=20)
rsi_upper = exp_rsi.number_input("RSI Upper", min_value=50, max_value=90, value=70)
rsi_lower = exp_rsi.number_input("RSI Lower", min_value=10, max_value=50, value=30)

# Main Content
st.title("A Simple Web App for Technical Analysis")
st.write("""
### User Manual:
- Select an S&P 500 stock
- Choose the time period of interest
- Download data as CSV
- Add technical indicators: SMA, Bollinger Bands, RSI
- Experiment with different parameters
""")

df = load_data(ticker, start_date, end_date)

# Data Preview
data_exp = st.expander("Preview Data")
available_cols = df.columns.tolist()
columns_to_show = data_exp.multiselect("Columns", available_cols, default=available_cols)
data_exp.dataframe(df[columns_to_show])

csv_file = convert_df_to_csv(df[columns_to_show])
data_exp.download_button(
    label="Download Selected as CSV",
    data=csv_file,
    file_name=f"{ticker}_stock_prices.csv",
    mime="text/csv",
)

# Create a 3-row subplot figure:
#  - Row 1: Price, SMA, Bollinger
#  - Row 2: RSI
#  - Row 3: Volume
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
    row_heights=[0.5, 0.25, 0.25]  # Adjust as desired
)

# 1) PRICE CHART (row=1)
fig.add_trace(
    go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"),
    row=1, col=1
)

# Add SMA if selected
if sma_flag:
    df["SMA"] = df["Close"].rolling(window=sma_periods).mean()
    fig.add_trace(
        go.Scatter(x=df.index, y=df["SMA"], mode="lines", name=f"SMA ({sma_periods})",
                   line=dict(color="orange")),
        row=1, col=1
    )

# Add Bollinger Bands if selected
if bb_flag:
    rolling_mean = df["Close"].rolling(window=bb_periods).mean()
    rolling_std = df["Close"].rolling(window=bb_periods).std()
    upper_band = rolling_mean + (bb_std * rolling_std)
    lower_band = rolling_mean - (bb_std * rolling_std)

    fig.add_trace(
        go.Scatter(x=df.index, y=upper_band, mode="lines", name="Upper BB",
                   line=dict(dash="dot", color="red")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=lower_band, mode="lines", name="Lower BB",
                   line=dict(dash="dot", color="green")),
        row=1, col=1
    )

# 2) RSI (row=2)
if rsi_flag:
    df["RSI"] = 100 - (100 / (1 + (
        df["Close"].diff().clip(lower=0).rolling(rsi_periods).mean() /
        df["Close"].diff().clip(upper=0).rolling(rsi_periods).mean().abs()
    )))
    fig.add_trace(
        go.Scatter(x=df.index, y=df["RSI"], mode="lines", name=f"RSI ({rsi_periods})",
                   line=dict(color="purple")),
        row=2, col=1
    )
    # Add horizontal lines for RSI thresholds
    fig.add_hline(
        y=rsi_upper, line=dict(color="gray", dash="dash"),
        row=2, col=1
    )
    fig.add_hline(
        y=rsi_lower, line=dict(color="gray", dash="dash"),
        row=2, col=1
    )

# 3) VOLUME (row=3)
if volume_flag and "Volume" in df.columns:
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume", opacity=0.6),
        row=3, col=1
    )

# Update Y-axes labels in each row
fig.update_yaxes(title_text="Price ($)", row=1, col=1)
if rsi_flag:
    fig.update_yaxes(title_text="RSI", row=2, col=1)
if volume_flag:
    fig.update_yaxes(title_text="Volume", row=3, col=1)

# Final Layout
fig.update_layout(
    title=f"{tickers_companies_dict[ticker]}'s Stock Price",
    template="plotly_white",
)

# Render the chart in Streamlit, with container width
st.plotly_chart(fig, use_container_width=True)
