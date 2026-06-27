"""
HSI MA Crossover "Buy Low Sell High" Dashboard
Interactive Website powered by Streamlit
Run with: streamlit run hsi_ma_dashboard.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(
    page_title="HSI MA Trading System | Buy Low Sell High",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional finance look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
    .stMetric {
        font-size: 1.1rem;
    }
    .analysis-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

def compute_max_drawdown(cum_returns):
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min() * 100

def fetch_and_analyze(ticker, start_date, end_date, short_window, long_window, rf=0.02):
    """Core analysis function from the original code, adapted for Streamlit"""
    
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if df.empty or len(df) < max(short_window, long_window) + 10:
        return None, "Insufficient data. Try a longer date range or different ticker."
    
    df = df[['Close']].dropna()
    df.columns = ['Close']
    
    # Big data summary
    start_price = df['Close'].iloc[0]
    end_price = df['Close'].iloc[-1]
    years = (df.index[-1] - df.index[0]).days / 365.25
    
    bh_total = (end_price / start_price - 1) * 100
    bh_cagr = ((end_price / start_price) ** (1 / years) - 1) * 100 if years > 0 else 0
    bh_vol = df['Close'].pct_change().std() * np.sqrt(252) * 100
    bh_max_dd = compute_max_drawdown((1 + df['Close'].pct_change()).cumprod())
    
    # Moving Averages
    df['SMA_Short'] = df['Close'].rolling(short_window).mean()
    df['SMA_Long'] = df['Close'].rolling(long_window).mean()
    
    # Signals
    df['Signal'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1, 0)
    df['Position'] = df['Signal'].shift(1).fillna(0)
    
    # Returns
    df['Market_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Position'] * df['Market_Return']
    
    df['Cum_Market'] = (1 + df['Market_Return']).cumprod()
    df['Cum_Strategy'] = (1 + df['Strategy_Return']).cumprod()
    
    # Strategy metrics
    strat_total = (df['Cum_Strategy'].iloc[-1] - 1) * 100
    strat_cagr = ((df['Cum_Strategy'].iloc[-1]) ** (1 / years) - 1) * 100 if years > 0 else 0
    strat_vol = df['Strategy_Return'].std() * np.sqrt(252) * 100
    strat_max_dd = compute_max_drawdown(df['Cum_Strategy'])
    
    excess = df['Strategy_Return'].dropna() - (rf / 252)
    sharpe = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0
    
    # Trade count
    trades = int(df['Position'].diff().abs().sum())
    
    # Current signal
    current_signal = "BUY / LONG 📈" if df['Signal'].iloc[-1] == 1 else "SELL / CASH 💵"
    
    metrics = {
        'bh_total': bh_total, 'bh_cagr': bh_cagr, 'bh_vol': bh_vol, 'bh_max_dd': bh_max_dd,
        'strat_total': strat_total, 'strat_cagr': strat_cagr, 'strat_vol': strat_vol,
        'strat_max_dd': strat_max_dd, 'sharpe': sharpe, 'trades': trades,
        'years': years, 'start_price': start_price, 'end_price': end_price,
        'current_signal': current_signal, 'last_date': df.index[-1].strftime('%Y-%m-%d')
    }
    
    return df, metrics

# ==================== UI ====================

st.title("📊 Hang Seng Index MA Trading System")
st.subheader("Interactive 'Buy Low, Sell High' Dashboard — Powered by 20 Years of Big Data Analysis")

st.markdown("""
**Based on the original Python MA crossover system.**  
This web app lets you analyze the Hang Seng Index (^HSI) and major constituent stocks with real historical data, 
adjustable moving averages, full backtesting, and professional visualizations.
""")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Strategy Parameters")
    
    ticker = st.selectbox(
        "Select Ticker",
        options=["^HSI", "0700.HK (Tencent)", "0005.HK (HSBC)", "1299.HK (AIA)", "9988.HK (Alibaba)", "1810.HK (Xiaomi)", "Custom"],
        index=0
    )
    
    if ticker == "Custom":
        ticker = st.text_input("Enter custom ticker (e.g. 0700.HK)", value="^HSI")
    else:
        ticker = ticker.split()[0]  # Extract ticker code
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2006, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    st.divider()
    
    short_window = st.slider("Short MA Window (days)", 10, 100, 50, step=5)
    long_window = st.slider("Long MA Window (days)", 50, 300, 200, step=10)
    
    if short_window >= long_window:
        st.warning("Short MA should be shorter than Long MA for crossover strategy.")
    
    st.divider()
    
    run_button = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)
    
    st.caption("Data source: Yahoo Finance | Analysis period up to 20+ years")

# Main content
if run_button:
    with st.spinner(f"Fetching data & running MA strategy for {ticker}..."):
        df, metrics = fetch_and_analyze(ticker, start_date.strftime("%Y-%m-%d"), 
                                        end_date.strftime("%Y-%m-%d"), 
                                        short_window, long_window)
    
    if df is None:
        st.error(metrics)
    else:
        # ========== HEADER METRICS ==========
        st.success(f"Analysis complete for **{ticker}** | {metrics['last_date']}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Signal", metrics['current_signal'], 
                     help="Based on latest SMA crossover")
        with col2:
            delta_cagr = metrics['strat_cagr'] - metrics['bh_cagr']
            st.metric("Strategy CAGR", f"{metrics['strat_cagr']:.2f}%", 
                     f"{delta_cagr:+.2f}% vs Buy&Hold")
        with col3:
            st.metric("Strategy Max Drawdown", f"{metrics['strat_max_dd']:.1f}%",
                     f"{metrics['bh_max_dd'] - metrics['strat_max_dd']:+.1f}% better" if metrics['strat_max_dd'] < metrics['bh_max_dd'] else "")
        with col4:
            st.metric("Sharpe Ratio (Strategy)", f"{metrics['sharpe']:.2f}")
        
        # ========== TABS ==========
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Price & Signals", 
            "💰 Performance Comparison", 
            "📊 Big Data Analysis", 
            "📋 Trade Log & Download"
        ])
        
        with tab1:
            st.subheader("Price Chart with Moving Averages & Buy/Sell Signals")
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.7, 0.3],
                               vertical_spacing=0.05)
            
            # Price + MAs
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', 
                                    line=dict(color='black', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Short'], name=f'SMA {short_window}', 
                                    line=dict(color='#1E88E5', width=1.8)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Long'], name=f'SMA {long_window}', 
                                    line=dict(color='#E53935', width=1.8)), row=1, col=1)
            
            # Buy/Sell markers
            buy_idx = df[(df['Position'].diff() == 1) & (df['Position'] == 1)].index
            sell_idx = df[(df['Position'].diff() == -1) & (df['Position'] == 0)].index
            
            if len(buy_idx) > 0:
                fig.add_trace(go.Scatter(x=buy_idx, y=df.loc[buy_idx, 'Close'],
                                        mode='markers', name='BUY Signal',
                                        marker=dict(symbol='triangle-up', size=12, color='green')), 
                             row=1, col=1)
            if len(sell_idx) > 0:
                fig.add_trace(go.Scatter(x=sell_idx, y=df.loc[sell_idx, 'Close'],
                                        mode='markers', name='SELL Signal',
                                        marker=dict(symbol='triangle-down', size=12, color='red')), 
                             row=1, col=1)
            
            # Position / Signal in lower pane
            fig.add_trace(go.Scatter(x=df.index, y=df['Position'], name='Position (1=Long, 0=Cash)',
                                    line=dict(color='#43A047', width=1.5), fill='tozeroy'), 
                         row=2, col=1)
            
            fig.update_layout(height=650, title_text=f"{ticker} — MA Crossover Strategy",
                             hovermode="x unified", legend=dict(orientation="h", y=1.05))
            fig.update_yaxes(title_text="Price (HKD)", row=1, col=1)
            fig.update_yaxes(title_text="Position", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Cumulative Return Comparison")
            
            perf_fig = go.Figure()
            perf_fig.add_trace(go.Scatter(x=df.index, y=df['Cum_Market'], 
                                         name='Buy & Hold', line=dict(color='#757575', width=2.5)))
            perf_fig.add_trace(go.Scatter(x=df.index, y=df['Cum_Strategy'], 
                                         name='MA Strategy', line=dict(color='#2E7D32', width=2.5)))
            
            perf_fig.update_layout(height=500, title="Cumulative Growth of $1 Invested",
                                  yaxis_title="Growth Multiple", xaxis_title="Date",
                                  hovermode="x unified")
            st.plotly_chart(perf_fig, use_container_width=True)
            
            # Key comparison table
            comparison = pd.DataFrame({
                'Metric': ['Total Return', 'CAGR', 'Volatility (ann.)', 'Max Drawdown', 'Sharpe Ratio'],
                'Buy & Hold': [f"{metrics['bh_total']:.1f}%", f"{metrics['bh_cagr']:.2f}%", 
                              f"{metrics['bh_vol']:.1f}%", f"{metrics['bh_max_dd']:.1f}%", "—"],
                'MA Strategy': [f"{metrics['strat_total']:.1f}%", f"{metrics['strat_cagr']:.2f}%", 
                               f"{metrics['strat_vol']:.1f}%", f"{metrics['strat_max_dd']:.1f}%", f"{metrics['sharpe']:.2f}"]
            })
            st.dataframe(comparison, use_container_width=True, hide_index=True)
        
        with tab3:
            st.subheader("20-Year Big Data Analysis Summary")
            
            st.markdown(f"""
            <div class="analysis-box">
            <h4>Key Insights for {ticker} ({metrics['years']:.1f} years)</h4>
            <ul>
                <li><strong>Price Return (Buy & Hold):</strong> {metrics['bh_total']:.1f}% total → CAGR {metrics['bh_cagr']:.2f}%</li>
                <li><strong>Strategy Improvement:</strong> MA crossover delivered {metrics['strat_total'] - metrics['bh_total']:+.1f}% extra total return</li>
                <li><strong>Risk Reduction:</strong> Max Drawdown improved by {metrics['bh_max_dd'] - metrics['strat_max_dd']:.1f} percentage points</li>
                <li><strong>Number of trades generated:</strong> {metrics['trades']}</li>
                <li><strong>Current Position:</strong> {metrics['current_signal']}</li>
            </ul>
            <p><em>Note: Results are price returns only (no dividends reinvested). Real total returns would be higher due to ~3% average dividend yield on HSI.</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("""
            **Historical Context (2006–2026):**  
            The Hang Seng Index delivered modest price returns over the past 20 years due to major crashes (2008: -48%, 2022-23 bear market) 
            and long sideways periods after 2018. The MA strategy helps avoid large drawdowns by moving to cash during downtrends, 
            at the cost of some whipsaw trades in ranging markets.
            """)
        
        with tab4:
            st.subheader("Trade Log & Data Export")
            
            # Simple trade log
            signal_changes = df[df['Position'].diff().abs() > 0][['Close', 'Position']].copy()
            signal_changes['Action'] = signal_changes['Position'].map({1: 'BUY', 0: 'SELL'})
            signal_changes = signal_changes.rename(columns={'Close': 'Price'})
            
            st.dataframe(signal_changes.tail(20), use_container_width=True)
            
            # Download buttons
            csv = df[['Close', 'SMA_Short', 'SMA_Long', 'Signal', 'Position', 'Cum_Strategy']].to_csv()
            st.download_button(
                label="📥 Download Full Analysis CSV",
                data=csv,
                file_name=f"{ticker}_ma_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            st.caption("The CSV includes price, moving averages, signals, and cumulative strategy performance.")

else:
    # Default landing / instructions
    st.info("👈 Adjust parameters in the sidebar and click **Run Full Analysis** to start.")
    
    st.markdown("""
    ### How this website works
    - Fetches real historical data from Yahoo Finance (20+ years where available)
    - Calculates Simple Moving Average crossover signals
    - Backtests the strategy vs simple Buy & Hold
    - Shows clear buy/sell signals and performance metrics
    - Fully interactive — change MA periods and instantly see impact
    
    **Typical good settings:**
    - Index (^HSI): 50 / 200
    - Individual stocks (more volatile): 30 / 100 or 20 / 50
    """)
    
    st.warning("⚠️ This is for educational and analytical purposes only. Past performance does not guarantee future results. Always do your own research.")

# Footer
st.divider()
st.caption("Built with the original MA crossover Python logic • Data: Yahoo Finance • For Hong Kong market analysis")