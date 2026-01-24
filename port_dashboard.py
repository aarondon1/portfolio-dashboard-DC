import time
import tempfile
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from alpaca.trading.client import TradingClient

try:
    import quantstats as qs
    qs.extend_pandas()
except Exception:
    qs = None

# Page configuration
st.set_page_config(
    page_title="Dolphin Capital – Portfolio Dashboard",
    page_icon="🐬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CONSTANTS ====================

CSV_PATH = "historical_positions_clean.csv"
CACHE_TTL = 300  # 5 minutes
YFINANCE_TIMEOUT = 10  # seconds

# ==================== ALPACA FUNCTIONS ====================


@st.cache_resource
def get_trading_client():
    """Initialize and cache the Alpaca TradingClient"""
    try:
        return TradingClient(
            st.secrets["ALPACA_KEY"],
            st.secrets["ALPACA_SECRET"],
            paper=str(st.secrets.get("ALPACA_PAPER", True)).lower() in ["true", "1", "yes"]
        )
    except Exception as e:
        st.error(f"⚠️ Alpaca credentials missing or invalid: {e}")
        return None


@st.cache_data(ttl=60)
def fetch_portfolio_data(_trading_client: TradingClient):
    """Fetch account and positions data from Alpaca"""
    if _trading_client is None:
        return None, pd.DataFrame()
    
    try:
        account = _trading_client.get_account()
        positions = _trading_client.get_all_positions()

        pos_df = pd.DataFrame([p.__dict__ for p in positions])

        if not pos_df.empty:
            for col in ["qty", "market_value", "avg_entry_price", "unrealized_pl", "unrealized_plpc"]:
                if col in pos_df.columns:
                    pos_df[col] = pd.to_numeric(pos_df[col], errors="coerce")

            mv_sum = pos_df["market_value"].sum()
            pos_df["weight"] = pos_df["market_value"] / mv_sum if mv_sum > 0 else 0.0

        return account, pos_df
    except Exception as e:
        st.error(f"Alpaca API error: {e}")
        return None, pd.DataFrame()


# ==================== HISTORICAL PORTFOLIO FUNCTIONS ====================


@st.cache_data(ttl=CACHE_TTL)
def load_historical_transactions(csv_path=CSV_PATH):
    """Load and process historical transactions from CSV"""
    try:
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Shares"] = pd.to_numeric(df["Shares"], errors="coerce")
        df["entry"] = pd.to_numeric(df["entry"], errors="coerce")
        df["Type"] = df["Type"].str.strip().str.lower()
        df["Ticker"] = df["Ticker"].str.strip().str.upper()
        
        # Ensure sell transactions have negative shares
        mask = df["Type"] == "sell"
        df.loc[mask, "Shares"] = -df.loc[mask, "Shares"].abs()
        
        return df.dropna(subset=["Date", "Ticker", "Shares"]).sort_values("Date").reset_index(drop=True)
    except FileNotFoundError:
        st.error(f"⚠️ CSV file not found: {csv_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"⚠️ Error loading CSV: {e}")
        return pd.DataFrame()


def calculate_current_positions(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate current positions from transaction history"""
    if transactions_df.empty:
        return pd.DataFrame()

    positions = []
    for ticker in transactions_df["Ticker"].unique():
        ticker_txns = transactions_df[transactions_df["Ticker"] == ticker].copy()
        net_shares = ticker_txns["Shares"].sum()

        if net_shares > 0.001:  # Only open positions
            buys = ticker_txns[ticker_txns["Shares"] > 0]
            avg_entry = (buys["Shares"] * buys["entry"]).sum() / buys["Shares"].sum() if not buys.empty else 0.0

            positions.append({
                "ticker": ticker,
                "shares": net_shares,
                "avg_entry_price": avg_entry,
            })

    return pd.DataFrame(positions)


@st.cache_data(ttl=60)
def get_current_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch current prices from yfinance (batched)"""
    if not tickers:
        return {}

    tickers = list(dict.fromkeys([t.upper() for t in tickers if t]))
    
    try:
        raw = yf.download(
            tickers,
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            group_by="column",
            timeout=YFINANCE_TIMEOUT
        )
    except Exception as e:
        st.warning(f"Price fetch error: {e}")
        return {t: None for t in tickers}

    if raw is None or raw.empty:
        return {t: None for t in tickers}

    prices = {}
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw.get("Close")
        if close is not None and not close.empty:
            last = close.ffill().iloc[-1]
            for t in tickers:
                prices[t] = float(last[t]) if t in last.index and pd.notna(last[t]) else None
    else:
        if "Close" in raw.columns and not raw["Close"].empty:
            prices[tickers[0]] = float(raw["Close"].dropna().iloc[-1])

    return prices


def enrich_positions_with_prices(positions_df: pd.DataFrame) -> pd.DataFrame:
    """Add current prices and calculate P&L"""
    if positions_df.empty:
        return positions_df

    positions_df = positions_df.copy()
    prices = get_current_prices(positions_df["ticker"].tolist())
    
    positions_df["current_price"] = positions_df["ticker"].map(prices)
    positions_df["market_value"] = positions_df["shares"] * positions_df["current_price"]
    positions_df["cost_basis"] = positions_df["shares"] * positions_df["avg_entry_price"]
    positions_df["unrealized_pl"] = positions_df["market_value"] - positions_df["cost_basis"]
    
    # Safe division for percentage
    positions_df["unrealized_plpc"] = 0.0
    mask = positions_df["cost_basis"] > 0
    positions_df.loc[mask, "unrealized_plpc"] = (
        positions_df.loc[mask, "unrealized_pl"] / positions_df.loc[mask, "cost_basis"] * 100
    )

    total = positions_df["market_value"].sum()
    positions_df["weight"] = positions_df["market_value"] / total if total > 0 else 0.0

    return positions_df


# ==================== ANALYTICS FUNCTIONS ====================


@st.cache_data(ttl=CACHE_TTL)
def compute_historical_portfolio_value(transactions_df: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.Series:
    """Build daily portfolio value from transaction history"""
    if transactions_df.empty:
        return pd.Series(dtype=float)

    txns = transactions_df[transactions_df["Date"] <= end_dt].copy()
    if txns.empty:
        return pd.Series(dtype=float)

    tickers = txns["Ticker"].unique().tolist()
    
    try:
        raw = yf.download(
            tickers,
            start=start_dt,
            end=end_dt,
            progress=False,
            auto_adjust=True,
            timeout=YFINANCE_TIMEOUT
        )
    except Exception as e:
        st.error(f"Failed to download price history: {e}")
        return pd.Series(dtype=float)

    if raw is None or raw.empty:
        return pd.Series(dtype=float)

    # Extract close prices
    if isinstance(raw, pd.Series):
        prices = raw.to_frame(name=tickers[0])
    elif isinstance(raw.columns, pd.MultiIndex):
        prices = raw.get("Close")
        if prices is None:
            prices = raw.get("Adj Close")
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw.get("Adj Close")

    if prices is None or prices.empty:
        return pd.Series(dtype=float)
    
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    
    prices.columns = [str(c).upper() for c in prices.columns]
    prices = prices.ffill()

    # Calculate daily portfolio value
    portfolio_values = []
    for date in prices.index:
        txns_to_date = txns[txns["Date"] <= date]
        holdings = txns_to_date.groupby("Ticker")["Shares"].sum()
        holdings = holdings[holdings > 0]

        value = sum(
            holdings.get(t, 0) * prices.loc[date, t]
            for t in holdings.index
            if t in prices.columns and pd.notna(prices.loc[date, t])
        )
        portfolio_values.append(value)

    return pd.Series(portfolio_values, index=prices.index, name="portfolio_value")


@st.cache_data(ttl=CACHE_TTL)
def get_benchmark_returns(benchmark: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.Series:
    """Fetch benchmark returns"""
    try:
        raw = yf.download(
            [benchmark],
            start=start_dt,
            end=end_dt,
            progress=False,
            auto_adjust=True,
            timeout=YFINANCE_TIMEOUT
        )
    except Exception as e:
        st.warning(f"Benchmark download failed: {e}")
        return pd.Series(dtype=float)
    
    if raw is None or raw.empty:
        return pd.Series(dtype=float)

    # Robust price extraction
    prices = None
    if isinstance(raw, pd.DataFrame):
        if "Close" in raw.columns:
            prices = raw["Close"]
        elif "Adj Close" in raw.columns:
            prices = raw["Adj Close"]
    else:
        prices = raw
    
    if prices is None or prices.empty:
        return pd.Series(dtype=float)
    
    # Convert to Series if needed
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] == 1:
            prices = prices.iloc[:, 0]
        else:
            return pd.Series(dtype=float)
    
    return prices.pct_change().dropna()


# ==================== DISPLAY FUNCTIONS ====================


def setup_refresh_controls():
    """Setup sidebar refresh controls"""
    st.sidebar.header("⚙️ Controls")
    refresh_mode = st.sidebar.selectbox("Refresh mode", ["Manual", "Every 15 sec", "Every 60 sec"])

    if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if refresh_mode == "Every 15 sec":
        time.sleep(15)
        st.rerun()
    elif refresh_mode == "Every 60 sec":
        time.sleep(60)
        st.rerun()

    return refresh_mode


def display_portfolio_kpis(total_value: float, cash: float, num_positions: int, total_pl: float = None):
    """Display portfolio KPIs"""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Total Value", f"${total_value:,.2f}")
    c2.metric("💵 Cash", f"${cash:,.2f}")
    c3.metric("📊 Equity", f"${(total_value + cash):,.2f}")
    c4.metric("📈 Positions", int(num_positions))

    if total_pl is not None:
        c5, _, _, _ = st.columns(4)
        cost_basis = total_value - total_pl
        delta = f"{(total_pl / cost_basis * 100):.2f}%" if cost_basis > 0 else None
        c5.metric("💹 Unrealized P&L", f"${total_pl:,.2f}", delta=delta)


def display_positions_table(pos_df: pd.DataFrame, is_historical: bool = False):
    """Display positions table"""
    if pos_df.empty:
        st.info("No open positions.")
        return

    symbol_col = "ticker" if is_historical else "symbol"
    qty_col = "shares" if is_historical else "qty"

    display_cols = [symbol_col, qty_col, "avg_entry_price", "market_value", "unrealized_pl", "unrealized_plpc", "weight"]
    display_df = pos_df[display_cols].copy()
    
    # Convert to percentages for display
    if is_historical:
        display_df["unrealized_plpc"] *= 100
    display_df["weight"] *= 100

    column_config = {
        symbol_col: st.column_config.TextColumn("Symbol", width="small"),
        qty_col: st.column_config.NumberColumn("Shares", format="%.2f"),
        "avg_entry_price": st.column_config.NumberColumn("Avg Entry", format="$%.2f"),
        "market_value": st.column_config.NumberColumn("Market Value", format="$%.2f"),
        "unrealized_pl": st.column_config.NumberColumn("P&L", format="$%.2f"),
        "unrealized_plpc": st.column_config.NumberColumn("P&L %", format="%.2f%%"),
        "weight": st.column_config.NumberColumn("Weight", format="%.2f%%"),
    }

    st.dataframe(display_df, use_container_width=True, hide_index=True, column_config=column_config)


def display_charts(pos_df: pd.DataFrame, is_historical: bool = False):
    """Display allocation and top positions charts"""
    if pos_df.empty:
        return

    symbol_col = "ticker" if is_historical else "symbol"

    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(
            pos_df,
            names=symbol_col,
            values="market_value",
            title="Portfolio Allocation",
            hole=0.4
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        top_n = min(10, len(pos_df))
        top_df = pos_df.nlargest(top_n, "market_value")
        fig_bar = px.bar(
            top_df,
            x=symbol_col,
            y="market_value",
            title=f"Top {top_n} Positions",
            color="unrealized_pl",
            color_continuous_scale=["red", "yellow", "green"],
            labels={"market_value": "Market Value ($)", symbol_col: "Symbol"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)


def display_portfolio_analytics(transactions_df: pd.DataFrame, current_equity: float = None):
    """Comprehensive analytics display with export options"""
    with st.expander("📈 Portfolio Analytics", expanded=False):
        if qs is None:
            st.warning("📦 Install QuantStats: `pip install quantstats`")
            return

        if transactions_df.empty:
            st.warning("No transaction history available.")
            return

        portfolio_start = transactions_df["Date"].min()

        # Controls
        c1, c2, c3 = st.columns([2, 2, 1])
        start_date = c1.date_input(
            "Start Date",
            value=portfolio_start.date(),
            min_value=portfolio_start.date(),
            max_value=datetime.now().date(),
            key="analytics_start"
        )
        end_date = c2.date_input(
            "End Date",
            value=datetime.now().date(),
            min_value=portfolio_start.date(),
            max_value=datetime.now().date(),
            key="analytics_end"
        )
        run = c3.button("Run Analysis", type="primary", use_container_width=True)

        if not run:
            st.caption(f"📅 Portfolio inception: {portfolio_start.date()}")
            return

        start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
        if start_dt < portfolio_start:
            st.error(f"Start date cannot be before portfolio inception ({portfolio_start.date()})")
            return
        if start_dt >= end_dt:
            st.error("Start date must be before end date.")
            return

        benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "IWM", "VTI"], index=0, key="benchmark")

        with st.spinner("Computing analytics..."):
            portfolio_values = compute_historical_portfolio_value(transactions_df, start_dt, end_dt)
            if portfolio_values.empty:
                st.error("No portfolio value computed. Check transaction history and date range.")
                return

            portfolio_returns = portfolio_values.pct_change().dropna()
            if portfolio_returns.empty:
                st.error("No returns computed.")
                return
            
            benchmark_returns = get_benchmark_returns(benchmark, start_dt, end_dt)

        # Align returns
        bench_ok = False
        if not benchmark_returns.empty:
            aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").dropna()
            if not aligned.empty and aligned.shape[1] == 2:
                portfolio_returns = aligned.iloc[:, 0]
                benchmark_returns = aligned.iloc[:, 1]
                bench_ok = benchmark_returns.var() > 1e-10

        # Metrics
        try:
            sharpe = qs.stats.sharpe(portfolio_returns)
            sortino = qs.stats.sortino(portfolio_returns)
            max_dd = qs.stats.max_drawdown(portfolio_returns)
            cagr = qs.stats.cagr(portfolio_returns)
            vol = qs.stats.volatility(portfolio_returns)
            calmar = qs.stats.calmar(portfolio_returns)
        except Exception as e:
            st.error(f"Metric calculation error: {e}")
            return
        
        var_95 = portfolio_returns.mean() - 1.65 * portfolio_returns.std()

        st.subheader("📊 Key Metrics")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Sharpe Ratio", f"{sharpe:.2f}")
        r2.metric("Sortino Ratio", f"{sortino:.2f}")
        r3.metric("Calmar Ratio", f"{calmar:.2f}")
        r4.metric("VaR 95% (1d)", f"{var_95:.2%}")

        r5, r6, r7, r8 = st.columns(4)
        r5.metric("Volatility (Ann.)", f"{vol:.2%}")
        r6.metric("Max Drawdown", f"{max_dd:.2%}")
        r7.metric("CAGR", f"{cagr:.2%}")
        
        if bench_ok:
            beta = portfolio_returns.cov(benchmark_returns) / benchmark_returns.var()
            r8.metric(f"Beta vs {benchmark}", f"{beta:.2f}")
        else:
            r8.metric(f"Beta vs {benchmark}", "N/A")

        # Charts
        st.subheader("📈 Portfolio Value Over Time")
        st.line_chart(portfolio_values, use_container_width=True)

        st.subheader("📈 Cumulative Returns")
        cum_returns = (1 + portfolio_returns).cumprod()
        if bench_ok:
            cum_bench = (1 + benchmark_returns).cumprod()
            chart_df = pd.DataFrame({
                "Portfolio": cum_returns,
                benchmark: cum_bench
            })
            st.line_chart(chart_df, use_container_width=True)
        else:
            st.line_chart(cum_returns, use_container_width=True)

        st.subheader("📅 Monthly Returns")
        try:
            monthly = qs.stats.monthly_returns(portfolio_returns)
            st.dataframe(monthly.style.format("{:.2%}"), use_container_width=True)
        except:
            st.info("Insufficient data for monthly returns table")

        st.subheader("📉 Drawdown")
        try:
            dd = qs.stats.to_drawdown_series(portfolio_returns)
            st.area_chart(dd, use_container_width=True, color="#FF4B4B")
        except:
            st.info("Insufficient data for drawdown chart")

        # Export
        st.divider()
        st.subheader("💾 Export Options")
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            # CSV Export
            csv_data = pd.DataFrame({
                "Date": portfolio_returns.index,
                "Portfolio_Returns": portfolio_returns.values,
                "Cumulative_Returns": cum_returns.values
            })
            st.download_button(
                label="📥 Download Returns (CSV)",
                data=csv_data.to_csv(index=False),
                file_name=f"dolphin_capital_returns_{start_date}_{end_date}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_exp2:
            # HTML Tearsheet
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp:
                    tmp_path = tmp.name
                
                if bench_ok:
                    qs.reports.html(portfolio_returns, benchmark=benchmark_returns, output=tmp_path)
                else:
                    qs.reports.html(portfolio_returns, output=tmp_path)
                
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    html_data = f.read()
                os.unlink(tmp_path)
                
                st.download_button(
                    label="📥 Download Tearsheet (HTML)",
                    data=html_data,
                    file_name=f"dolphin_capital_tearsheet_{start_date}_{end_date}.html",
                    mime="text/html",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Tearsheet generation failed: {e}")


# ==================== VIEW FUNCTIONS ====================


def show_alpaca_portfolio():
    """Display Alpaca live portfolio"""
    st.header("📡 Live Alpaca Portfolio")

    client = get_trading_client()
    if client is None:
        st.error("❌ Cannot connect to Alpaca. Check your API credentials in secrets.")
        return

    account, pos_df = fetch_portfolio_data(client)
    if account is None:
        st.error("❌ Failed to fetch Alpaca portfolio data")
        return

    st.divider()
    display_portfolio_kpis(float(account.portfolio_value), float(account.cash), len(pos_df))
    
    if not pos_df.empty:
        st.divider()
        st.subheader("📋 Current Positions")
        display_positions_table(pos_df)
        st.divider()
        display_charts(pos_df)
    else:
        st.info("No open positions in Alpaca account")


def show_historical_portfolio():
    """Display historical CSV portfolio"""
    st.header("📜 Historical Portfolio")

    transactions = load_historical_transactions()
    if transactions.empty:
        st.warning("No historical transactions found.")
        return

    positions = calculate_current_positions(transactions)
    if positions.empty:
        st.info("No current positions (all closed)")
        st.divider()
        st.subheader("📊 Transaction History")
        st.dataframe(transactions.sort_values("Date", ascending=False), use_container_width=True)
        return

    positions = enrich_positions_with_prices(positions)
    total_value = positions["market_value"].sum()
    total_pl = positions["unrealized_pl"].sum()

    st.divider()
    display_portfolio_kpis(total_value, 0, len(positions), total_pl)
    st.divider()
    st.subheader("📋 Current Positions")
    display_positions_table(positions, is_historical=True)
    st.divider()
    display_charts(positions, is_historical=True)
    st.divider()
    st.subheader("📊 Transaction History")
    st.dataframe(transactions.sort_values("Date", ascending=False), use_container_width=True)


def show_combined_portfolio():
    """Display combined portfolio with analytics"""
    st.header("🔗 Combined Portfolio View")

    transactions = load_historical_transactions()
    alpaca_cash = 0.0
    combined_rows = []

    # Fetch Alpaca data
    client = get_trading_client()
    if client:
        account, alpaca_pos = fetch_portfolio_data(client)
        if account:
            alpaca_cash = float(account.cash)
            if not alpaca_pos.empty:
                alpaca_data = alpaca_pos[["symbol", "qty", "market_value", "unrealized_pl"]].rename(
                    columns={"symbol": "ticker", "qty": "shares"}
                ).assign(source="Alpaca")
                combined_rows.extend(alpaca_data.to_dict("records"))

    # Fetch historical data
    if not transactions.empty:
        hist_pos = enrich_positions_with_prices(calculate_current_positions(transactions))
        if not hist_pos.empty:
            hist_data = hist_pos[["ticker", "shares", "market_value", "unrealized_pl"]].assign(source="Historical")
            combined_rows.extend(hist_data.to_dict("records"))

    if not combined_rows:
        st.info("No positions in either portfolio")
        display_portfolio_kpis(0, alpaca_cash, 0, 0)
        return

    # Aggregate positions
    aggregated = pd.DataFrame(combined_rows).groupby("ticker", as_index=False).agg({
        "shares": "sum",
        "market_value": "sum",
        "unrealized_pl": "sum"
    }).sort_values("market_value", ascending=False)

    total_value = aggregated["market_value"].sum()
    total_pl = aggregated["unrealized_pl"].sum()
    aggregated["weight"] = aggregated["market_value"] / total_value if total_value > 0 else 0.0

    st.divider()
    display_portfolio_kpis(total_value, alpaca_cash, len(aggregated), total_pl)

    # Concentration metrics
    st.divider()
    st.subheader("📊 Concentration Metrics")
    weights = aggregated.set_index("ticker")["weight"]
    hhi = (weights ** 2).sum()
    
    m1, m2, m3 = st.columns(3)
    m1.metric("HHI (Herfindahl Index)", f"{hhi:.3f}", help="Lower = more diversified")
    m2.metric("Effective # of Positions", f"{1/hhi:.1f}", help="Portfolio behaves like N equal-weight positions")
    m3.metric("Top-5 Concentration", f"{weights.nlargest(5).sum():.2%}", help="Weight of top 5 holdings")

    st.divider()
    st.subheader("📋 Combined Holdings")
    display_df = aggregated.copy()
    display_df["weight"] = display_df["weight"] * 100
    
    column_config = {
        "ticker": st.column_config.TextColumn("Symbol", width="small"),
        "shares": st.column_config.NumberColumn("Shares", format="%.2f"),
        "market_value": st.column_config.NumberColumn("Market Value", format="$%.2f"),
        "unrealized_pl": st.column_config.NumberColumn("P&L", format="$%.2f"),
        "weight": st.column_config.NumberColumn("Weight", format="%.2f%%"),
    }
    st.dataframe(display_df, use_container_width=True, hide_index=True, column_config=column_config)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(aggregated, names="ticker", values="market_value", title="Combined Allocation", hole=0.4)
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        top10 = aggregated.nlargest(10, "market_value")
        fig_bar = px.bar(top10, x="ticker", y="market_value", title="Top 10 Positions",
                        color="unrealized_pl", color_continuous_scale=["red", "yellow", "green"])
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    display_portfolio_analytics(transactions, total_value + alpaca_cash)


def main():
    """Main application"""
    refresh_mode = setup_refresh_controls()

    st.title("🐬 Dolphin Capital – Portfolio Dashboard")
    st.caption(f"Last updated: {datetime.now():%Y-%m-%d %H:%M:%S} | Mode: {refresh_mode}")

    tab1, tab2, tab3 = st.tabs(["🔗 Combined View", "📡 Live Alpaca", "📜 Historical CSV"])

    with tab1:
        show_combined_portfolio()
    with tab2:
        show_alpaca_portfolio()
    with tab3:
        show_historical_portfolio()

    st.divider()
    with st.expander("About"):
        st.markdown("""
        **Dolphin Capital Portfolio Dashboard**
        
        - **Combined View**: Aggregated positions from both Alpaca live trading and historical CSV
        - **Live Alpaca**: Real-time positions from your Alpaca brokerage account
        - **Historical CSV**: Positions tracked via `historical_positions_clean.csv`
        
        Built with [Streamlit](https://streamlit.io) • [Alpaca Markets](https://alpaca.markets) • [yfinance](https://github.com/ranaroussi/yfinance) • [QuantStats](https://github.com/ranaroussi/quantstats)
        created by Aaron Don (this is still a beta version will constantly be updating, if any issues occur please dont hesitate to report them to aarondon004@gmail.com).
        """)


if __name__ == "__main__":
    main()