import dash
from dash import dcc, html, dash_table
import pandas as pd
import yfinance as yf
import datetime
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

###################
# Data Functions
###################
def get_sp500_components():
    """
    Returns a list of S&P 500 ticker symbols and a dictionary {ticker: company_name}.
    """
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    tickers = df["Symbol"].to_list()
    tickers_companies_dict = dict(zip(df["Symbol"], df["Security"]))
    return tickers, tickers_companies_dict

def load_data(symbol, start, end):
    """
    Fetch stock price data using yfinance.
    """
    return yf.download(symbol, start=start, end=end)

###################
# Build the Dash App
###################
app = dash.Dash(__name__)

# Get the S&P 500 tickers/companies
tickers_list, tickers_dict = get_sp500_components()

# Layout
app.layout = html.Div([
    html.H1("Technical Analysis Dashboard (Plotly Dash)"),
    
    # Ticker Dropdown
    html.Div([
        html.Label("Select Ticker:"),
        dcc.Dropdown(
            id="ticker-dropdown",
            options=[{"label": f"{tickers_dict[t]} ({t})", "value": t} for t in tickers_list],
            value="AAPL",  # Default selection
            clearable=False,
            style={"width": "300px"}
        )
    ], style={"margin-bottom": "20px"}),

    # Date Pickers
    html.Div([
        html.Label("Start Date:"),
        dcc.DatePickerSingle(
            id="start-date-picker",
            date=datetime.date(2019, 1, 1)
        ),
        html.Label("End Date:", style={"marginLeft": "20px"}),
        dcc.DatePickerSingle(
            id="end-date-picker",
            date=datetime.date.today()
        )
    ], style={"margin-bottom": "20px"}),

    # Checklist for Technical Indicators
    html.Div([
        html.Label("Technical Indicators:"),
        dcc.Checklist(
            id="tech-indicators",
            options=[
                {"label": "Add Volume", "value": "volume"},
                {"label": "Add SMA",    "value": "sma"},
                {"label": "Add Bollinger Bands", "value": "bb"},
                {"label": "Add RSI",    "value": "rsi"}
            ],
            value=[],  # Default: none selected
            inline=True
        )
    ], style={"margin-bottom": "20px"}),

    # Button to Load Data & Update Chart
    html.Button("Load / Update Chart", id="load-button", n_clicks=0),

    # Graph
    dcc.Graph(id="stock-graph"),

    # Data Table
    html.H3("Data Preview"),
    dash_table.DataTable(
        id="data-table",
        columns=[],
        data=[],
        page_size=10  # Show 10 rows at a time
    ),

    # Download CSV
    html.Button("Download CSV", id="download-btn", n_clicks=0, style={"marginTop": "20px"}),
    dcc.Download(id="download-component")
])

###################
# Callbacks
###################

@app.callback(
    [Output("stock-graph", "figure"),
     Output("data-table", "columns"),
     Output("data-table", "data")],
    [Input("load-button", "n_clicks")],
    [
     State("ticker-dropdown", "value"),
     State("start-date-picker", "date"),
     State("end-date-picker", "date"),
     State("tech-indicators", "value")
    ]
)
def update_chart(n_clicks, ticker, start_date, end_date, indicators):
    """
    Load the data from yfinance and build the Plotly figure with selected indicators.
    Also update the data preview table.
    """
    if not ticker or not start_date or not end_date:
        # If no selection, do nothing
        return go.Figure(), [], []

    # Convert the date strings to datetime.date if needed
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
    if start_date > end_date:
        # Basic guard: don't load if dates are invalid
        return go.Figure(), [], []

    # Load the data
    df = load_data(ticker, start_date, end_date)
    if df.empty:
        return go.Figure(), [], []

    # Build the base figure with the Closing Price
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df["Close"],
        mode="lines", 
        name="Close"
    ))

    # If "Add Volume" is selected
    if "volume" in indicators:
        # Add a secondary y-axis for volume
        fig.add_trace(go.Bar(
            x=df.index, 
            y=df["Volume"], 
            name="Volume", 
            opacity=0.3,
            yaxis="y2"
        ))
        fig.update_layout(
            yaxis2=dict(
                overlaying="y",
                side="right",
                showgrid=False,
                position=1
            )
        )

    # If "Add SMA" is selected
    if "sma" in indicators:
        df["SMA"] = df["Close"].rolling(window=20).mean()
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df["SMA"], 
            mode="lines", 
            name="SMA (20)",
            line=dict(color="orange")
        ))

    # If "Add Bollinger Bands" is selected
    if "bb" in indicators:
        rolling_mean = df["Close"].rolling(window=20).mean()
        rolling_std = df["Close"].rolling(window=20).std()
        upper_band = rolling_mean + 2 * rolling_std
        lower_band = rolling_mean - 2 * rolling_std

        fig.add_trace(go.Scatter(
            x=df.index, 
            y=upper_band, 
            mode="lines", 
            name="Upper BB",
            line=dict(dash="dot", color="red")
        ))
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=lower_band, 
            mode="lines", 
            name="Lower BB",
            line=dict(dash="dot", color="green")
        ))

    # If "Add RSI" is selected
    if "rsi" in indicators:
        # Basic RSI calculation
        delta = df["Close"].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / roll_down
        df["RSI"] = 100 - (100 / (1 + rs))

        # Add RSI as a separate trace (overlays by default)
        # You might prefer a separate figure or a secondary axis
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df["RSI"], 
            mode="lines", 
            name="RSI (14)",
            line=dict(color="purple")
        ))

    # Update figure layout
    fig.update_layout(
        title=f"{ticker} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        legend=dict(x=0, y=1)
    )

    # Data for the preview table
    # Convert index to column for a cleaner display
    df_reset = df.reset_index()
    columns = [{"name": col, "id": col} for col in df_reset.columns]
    data = df_reset.to_dict("records")

    return fig, columns, data

@app.callback(
    Output("download-component", "data"),
    Input("download-btn", "n_clicks"),
    [
     State("ticker-dropdown", "value"),
     State("start-date-picker", "date"),
     State("end-date-picker", "date")
    ]
)
def download_csv(n_clicks, ticker, start_date, end_date):
    """
    Provide a downloadable CSV of the data.
    """
    if not n_clicks:
        # Don't trigger until button is clicked
        return dash.no_update

    # Load the data again
    df = load_data(ticker, pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date())
    return dcc.send_data_frame(df.to_csv, f"{ticker}_stock_data.csv")

###################
# Run the app
###################
if __name__ == "__main__":
    # By default, Dash runs on http://127.0.0.1:8050
    app.run_server(debug=True)
