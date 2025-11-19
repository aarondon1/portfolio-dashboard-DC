import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from alpaca.trading.client import TradingClient
from alpaca.trading.models import Position
from datetime import datetime
import time
import yfinance as yf

# Page configuration
st.set_page_config(page_title="Dolphin Capital – Live Portfolio", layout="wide")

# ==================== ALPACA FUNCTIONS ====================

@st.cache_resource
def get_trading_client():
    """Initialize and cache the Alpaca TradingClient"""
    key = st.secrets["ALPACA_KEY"]
    secret = st.secrets["ALPACA_SECRET"]
    paper_flag = st.secrets.get("ALPACA_PAPER", True)
    
    if isinstance(paper_flag, str):
        paper_flag = paper_flag.lower() in ["true", "1", "yes"]
    
    return TradingClient(key, secret, paper=paper_flag)

def fetch_portfolio_data(trading_client):
    """Fetch account and positions data from Alpaca"""
    account = trading_client.get_account()
    positions = trading_client.get_all_positions()
    
    pos_raw = [p.__dict__ for p in positions]
    pos_df = pd.DataFrame(pos_raw)
    
    if not pos_df.empty:
        for col in ["qty", "market_value", "avg_entry_price", 
                    "unrealized_pl", "unrealized_plpc"]:
            if col in pos_df.columns:
                pos_df[col] = pd.to_numeric(pos_df[col], errors="coerce")
        
        pos_df["weight"] = pos_df["market_value"] / pos_df["market_value"].sum()
    
    return account, pos_df

# ==================== HISTORICAL PORTFOLIO FUNCTIONS ====================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_historical_transactions(csv_path="historical_positions_clean.csv"):
    """Load and process historical transactions from CSV"""
    try:
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Shares'] = pd.to_numeric(df['Shares'], errors='coerce')
        df['entry'] = pd.to_numeric(df['entry'], errors='coerce')
        df = df.dropna(subset=['Date', 'Ticker', 'Type', 'Shares'])
        return df
    except FileNotFoundError:
        st.error("⚠️ Historical positions CSV file not found!")
        return pd.DataFrame()

def calculate_current_positions(transactions_df):
    """Calculate current positions from transaction history"""
    if transactions_df.empty:
        return pd.DataFrame()
    
    # Group by ticker and calculate net shares
    positions = []
    
    for ticker in transactions_df['Ticker'].unique():
        ticker_txns = transactions_df[transactions_df['Ticker'] == ticker].copy()
        
        # Calculate shares (Buy = positive, Sell = negative)
        ticker_txns['signed_shares'] = ticker_txns.apply(
            lambda row: row['Shares'] if row['Type'] == 'Buy' else -abs(row['Shares']),
            axis=1
        )
        
        net_shares = ticker_txns['signed_shares'].sum()
        
        # Only include if we still hold shares
        if net_shares > 0:
            # Calculate average entry price (weighted by shares bought)
            buys = ticker_txns[ticker_txns['Type'] == 'Buy']
            if not buys.empty:
                total_cost = (buys['Shares'] * buys['entry']).sum()
                total_shares_bought = buys['Shares'].sum()
                avg_entry = total_cost / total_shares_bought if total_shares_bought > 0 else 0
            else:
                avg_entry = 0
            
            positions.append({
                'ticker': ticker,
                'shares': net_shares,
                'avg_entry_price': avg_entry
            })
    
    return pd.DataFrame(positions)

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_current_prices(tickers):
    """Fetch current prices from yfinance"""
    prices = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if not hist.empty:
                prices[ticker] = hist['Close'].iloc[-1]
            else:
                prices[ticker] = None
        except:
            prices[ticker] = None
    return prices

@st.cache_data(ttl=300)
def get_price_history(ticker, period="6mo", interval="1d"):
    """Fetch historical prices for a ticker"""
    try:
        history = yf.Ticker(ticker).history(period=period, interval=interval)
    except Exception:
        return pd.DataFrame()
    
    if history.empty:
        return history
    
    history = history.reset_index()
    return history
def enrich_positions_with_prices(positions_df):
    """Add current prices and calculate P&L"""
    if positions_df.empty:
        return positions_df
    
    current_prices = get_current_prices(positions_df['ticker'].tolist())
    
    positions_df['current_price'] = positions_df['ticker'].map(current_prices)
    positions_df['market_value'] = positions_df['shares'] * positions_df['current_price']
    positions_df['cost_basis'] = positions_df['shares'] * positions_df['avg_entry_price']
    positions_df['unrealized_pl'] = positions_df['market_value'] - positions_df['cost_basis']
    positions_df['unrealized_plpc'] = (positions_df['unrealized_pl'] / positions_df['cost_basis']) * 100
    
    # Calculate weights
    total_value = positions_df['market_value'].sum()
    positions_df['weight'] = positions_df['market_value'] / total_value
    
    return positions_df

# ==================== DISPLAY FUNCTIONS ====================

def setup_refresh_controls():
    """Setup sidebar refresh controls"""
    st.sidebar.header("⚙️ Controls")
    
    refresh_mode = st.sidebar.selectbox(
        "Refresh mode",
        ["Manual", "Every 15 sec", "Every 60 sec"]
    )
    
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()
    
    if refresh_mode == "Every 15 sec":
        time.sleep(15)
        st.rerun()
    elif refresh_mode == "Every 60 sec":
        time.sleep(60)
        st.rerun()
    
    return refresh_mode

def display_portfolio_kpis(total_value, cash, num_positions, total_pl=None):
    """Display portfolio KPIs"""
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric("💰 Total Value", f"${total_value:,.2f}")
    with kpi2:
        st.metric("💵 Cash", f"${cash:,.2f}")
    with kpi3:
        st.metric("📊 Equity", f"${total_value + cash:,.2f}")
    with kpi4:
        st.metric("📈 Open Positions", num_positions)
    
    if total_pl is not None:
        kpi5, kpi6, kpi7, kpi8 = st.columns(4)
        with kpi5:
            st.metric("💹 Unrealized P&L", f"${total_pl:,.2f}", 
                     delta=f"{(total_pl / (total_value - total_pl) * 100):.2f}%" if total_value > total_pl else "0%")

def display_positions_table(pos_df, is_historical=False):
    """Display positions table"""
    st.subheader("📋 Current Holdings")
    
    if pos_df.empty:
        st.info("No open positions.")
        return
    
    # Prepare display dataframe
    if is_historical:
        display_df = pos_df[[
            'ticker', 'shares', 'avg_entry_price', 'current_price',
            'market_value', 'unrealized_pl', 'unrealized_plpc', 'weight'
        ]].copy()
        
        column_config = {
            "ticker": "Symbol",
            "shares": "Shares",
            "avg_entry_price": st.column_config.NumberColumn("Avg Entry", format="$%.2f"),
            "current_price": st.column_config.NumberColumn("Current Price", format="$%.2f"),
            "market_value": st.column_config.NumberColumn("Market Value", format="$%.2f"),
            "unrealized_pl": st.column_config.NumberColumn("Unrealized P&L", format="$%.2f"),
            "unrealized_plpc": st.column_config.NumberColumn("P&L %", format="%.2f%%"),
            "weight": st.column_config.NumberColumn("Weight", format="%.2f%%")
        }
        
        display_df['weight'] = display_df['weight'] * 100
        
    else:
        display_df = pos_df[[
            "symbol", "qty", "market_value", "avg_entry_price",
            "unrealized_pl", "unrealized_plpc", "weight"
        ]].copy()
        
        display_df['unrealized_plpc'] = display_df['unrealized_plpc'] * 100
        display_df['weight'] = display_df['weight'] * 100
        
        column_config = {
            "symbol": "Symbol",
            "qty": "Quantity",
            "market_value": st.column_config.NumberColumn("Market Value", format="$%.2f"),
            "avg_entry_price": st.column_config.NumberColumn("Avg Entry Price", format="$%.2f"),
            "unrealized_pl": st.column_config.NumberColumn("Unrealized P&L", format="$%.2f"),
            "unrealized_plpc": st.column_config.NumberColumn("P&L %", format="%.2f%%"),
            "weight": st.column_config.NumberColumn("Weight", format="%.2f%%")
        }
    
    st.dataframe(
        display_df,
        width="stretch",  # Changed from use_container_width=True
        column_config=column_config,
        hide_index=True
    )

def display_allocation_chart(pos_df, is_historical=False):
    """Display allocation pie chart"""
    st.subheader("🥧 Asset Allocation")
    
    if pos_df.empty:
        st.info("No positions to display.")
        return
    
    if is_historical:
        chart_df = pos_df[['ticker', 'market_value']].copy()
        chart_df.columns = ['symbol', 'market_value']
    else:
        chart_df = pos_df[['symbol', 'market_value']].copy()
    
    fig_pie = px.pie(
        chart_df,
        names="symbol",
        values="market_value",
        hole=0.4,
        title="Portfolio Allocation"
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    
    st.plotly_chart(fig_pie, width="stretch")  # Changed from use_container_width=True

def display_top_positions_chart(pos_df, is_historical=False):
    """Display top positions bar chart"""
    st.subheader("📊 Top Positions by Value")
    
    if pos_df.empty:
        st.info("No positions to display.")
        return
    
    num_positions = len(pos_df)
    
    if num_positions <= 3:
        top_n = num_positions
        st.info(f"Showing all {num_positions} position(s)")
    else:
        max_positions = min(num_positions, 20)
        top_n = st.slider("Show top N holdings", min_value=3, max_value=max_positions, value=min(10, max_positions))
    
    if is_historical:
        top_df = pos_df.nlargest(top_n, 'market_value')
        symbol_col = 'ticker'
    else:
        top_df = pos_df.nlargest(top_n, 'market_value')
        symbol_col = 'symbol'
    
    fig_bar = px.bar(
        top_df,
        x=symbol_col,
        y="market_value",
        text="market_value",
        title=f"Top {top_n} Position{'s' if top_n != 1 else ''}",
        color="unrealized_pl",
        color_continuous_scale=["red", "yellow", "green"]
    )
    fig_bar.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    fig_bar.update_layout(
        xaxis_title="Symbol",
        yaxis_title="Market Value ($)",
        showlegend=False
    )
    
    st.plotly_chart(fig_bar, width="stretch")  # Changed from use_container_width=True

# ==================== MAIN APPLICATION ====================

def show_alpaca_portfolio():
    """Display Alpaca live portfolio"""
    st.header("📡 Live Alpaca Portfolio")
    
    try:
        trading_client = get_trading_client()
    except Exception as e:
        st.error(f"❌ Error initializing Alpaca client: {e}")
        return
    
    try:
        account, pos_df = fetch_portfolio_data(trading_client)
    except Exception as e:
        st.error(f"❌ Error fetching portfolio data: {e}")
        return
    
    st.divider()
    display_portfolio_kpis(
        total_value=float(account.portfolio_value),
        cash=float(account.cash),
        num_positions=len(pos_df)
    )
    
    st.divider()
    display_positions_table(pos_df, is_historical=False)
    
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        display_allocation_chart(pos_df, is_historical=False)
    with col2:
        display_top_positions_chart(pos_df, is_historical=False)

def show_historical_portfolio():
    """Display historical CSV-based portfolio"""
    st.header("📜 Historical Portfolio (CSV + yfinance)")
    
    transactions = load_historical_transactions()
    
    if transactions.empty:
        st.warning("No historical transactions found. Please add 'historical_positions.csv'")
        return
    
    positions = calculate_current_positions(transactions)
    
    if positions.empty:
        st.info("No current positions (all positions have been closed)")
        return
    
    positions = enrich_positions_with_prices(positions)
    
    total_value = positions['market_value'].sum()
    total_pl = positions['unrealized_pl'].sum()
    
    st.divider()
    display_portfolio_kpis(
        total_value=total_value,
        cash=0,  # No cash tracking in CSV
        num_positions=len(positions),
        total_pl=total_pl
    )
    
    st.divider()
    display_positions_table(positions, is_historical=True)
    
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        display_allocation_chart(positions, is_historical=True)
    with col2:
        display_top_positions_chart(positions, is_historical=True)
    
    # Show transaction history
    st.divider()
    st.subheader("📊 Transaction History")
    st.dataframe(transactions.sort_values('Date', ascending=False), width="stretch")  # Changed from use_container_width=True

def show_combined_portfolio():
    """Display combined view of both portfolios"""
    st.header("🔗 Combined Portfolio View")
    
    # Fetch both portfolios
    alpaca_pos = pd.DataFrame()
    historical_pos = pd.DataFrame()
    
    try:
        trading_client = get_trading_client()
        account, alpaca_pos = fetch_portfolio_data(trading_client)
        alpaca_cash = float(account.cash)
    except:
        alpaca_cash = 0
        st.warning("Could not load Alpaca portfolio")
    
    try:
        transactions = load_historical_transactions()
        if not transactions.empty:
            historical_pos = calculate_current_positions(transactions)
            historical_pos = enrich_positions_with_prices(historical_pos)
    except:
        st.warning("Could not load historical portfolio")
    
    # Combine positions
    combined = []
    
    if not alpaca_pos.empty:
        for _, row in alpaca_pos.iterrows():
            combined.append({
                'symbol': row['symbol'],
                'shares': float(row['qty']),
                'market_value': float(row['market_value']),
                'unrealized_pl': float(row['unrealized_pl']),
                'source': 'Alpaca'
            })
    
    if not historical_pos.empty:
        for _, row in historical_pos.iterrows():
            combined.append({
                'symbol': row['ticker'],
                'shares': row['shares'],
                'market_value': row['market_value'],
                'unrealized_pl': row['unrealized_pl'],
                'source': 'Historical'
            })
    
    if not combined:
        st.info("No positions in either portfolio")
        return
    
    combined_df = pd.DataFrame(combined)
    
    # Aggregate by symbol
    aggregated = combined_df.groupby('symbol').agg({
        'shares': 'sum',
        'market_value': 'sum',
        'unrealized_pl': 'sum'
    }).reset_index()
    
    total_value = aggregated['market_value'].sum()
    total_pl = aggregated['unrealized_pl'].sum()
    
    st.divider()
    display_portfolio_kpis(
        total_value=total_value,
        cash=alpaca_cash,
        num_positions=len(aggregated),
        total_pl=total_pl
    )
    
    st.divider()
    st.subheader("📋 Combined Holdings")
    st.dataframe(aggregated, width="stretch")  # Changed from use_container_width=True
    
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(aggregated, names='symbol', values='market_value', title="Combined Allocation")
        st.plotly_chart(fig_pie, width="stretch")  # Changed from use_container_width=True
    
    with col2:
        fig_bar = px.bar(aggregated.nlargest(10, 'market_value'), 
                        x='symbol', y='market_value', title="Top 10 Positions")
        st.plotly_chart(fig_bar, width="stretch")  # Changed from use_container_width=True

    st.divider()
    st.subheader("📈 Holding Price History")
    
    if aggregated.empty:
        st.info("No holdings available for price history.")
        return
    
    selection_col, range_col = st.columns([2, 1])
    
    with selection_col:
        selected_symbol = st.selectbox(
            "Select a holding",
            sorted(aggregated['symbol'].unique().tolist())
        )
    
    with range_col:
        range_label = st.selectbox(
            "Range",
            ["1M", "3M", "6M", "YTD", "1Y", "5Y", "Max"],
            index=2
        )

    chart_col, option_col = st.columns([1, 1])
    
    with chart_col:
        chart_type = st.selectbox(
            "Chart type",
            ["Line", "Candlestick"],
            index=1
        )
    
    with option_col:
        show_ma = st.checkbox("Show 20 & 50-day MA", value=True)
    
    period_map = {
        "1M": "1mo",
        "3M": "3mo",
        "6M": "6mo",
        "YTD": "ytd",
        "1Y": "1y",
        "5Y": "5y",
        "Max": "max"
    }
    
    history = get_price_history(selected_symbol, period=period_map[range_label])
    
    if history.empty:
        st.warning("No historical price data available for this symbol.")
        return
    
    history = history.copy()
    
    if show_ma and not history.empty:
        history["MA20"] = history["Close"].rolling(window=20).mean()
        history["MA50"] = history["Close"].rolling(window=50).mean()
    else:
        history["MA20"] = None
        history["MA50"] = None
    
    if chart_type == "Candlestick":
        fig_history = go.Figure(
            data=[
                go.Candlestick(
                    x=history["Date"],
                    open=history["Open"],
                    high=history["High"],
                    low=history["Low"],
                    close=history["Close"],
                    name="Price"
                )
            ]
        )
        fig_history.update_layout(
            title=f"{selected_symbol} candlestick ({range_label})",
            yaxis_title="Price ($)",
            xaxis_rangeslider_visible=False
        )
    else:
        fig_history = px.line(
            history,
            x="Date",
            y="Close",
            title=f"{selected_symbol} price history ({range_label})"
        )
        fig_history.update_yaxes(title="Price ($)")
    
    if show_ma:
        for ma_col, label, color in [
            ("MA20", "20-day MA", "#FFB347"),
            ("MA50", "50-day MA", "#2E86C1")
        ]:
            if history[ma_col].notna().any():
                fig_history.add_trace(
                    go.Scatter(
                        x=history["Date"],
                        y=history[ma_col],
                        mode="lines",
                        name=label,
                        line=dict(width=1.5, color=color)
                    )
                )
    
    st.plotly_chart(fig_history, width="stretch")

def main():
    """Main application"""
    refresh_mode = setup_refresh_controls()
    
    st.title("🐬 Dolphin Capital – Portfolio Dashboard")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Mode: {refresh_mode}")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔗 Combined View", "📡 Live Alpaca", "📜 Historical CSV"])
    
    with tab1:
        show_combined_portfolio()
    
    with tab2:
        show_alpaca_portfolio()
    
    with tab3:
        show_historical_portfolio()
    
    st.divider()
    st.caption("Built with Streamlit + Alpaca API + yfinance | Dolphin Capital")

if __name__ == "__main__":
    main()