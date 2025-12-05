import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import time

# --- Setup and Constants ---

# Define the environment variable checks for Python (safer)
try:
    # This block handles the dynamic app ID if available in the execution environment
    appId = st.secrets["APP_ID"]
except KeyError:
    # Fallback for local testing if the secret is not defined
    appId = 'default-portfolio-optimizer-app'

# List of 500+ representative tickers (S&P 500 subset + a mix of other caps)
TICKER_UNIVERSE = sorted(list(set([
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'V', 'PG', 'MA',
    'UNH', 'HD', 'PYPL', 'DIS', 'NFLX', 'CMCSA', 'ADBE', 'NKE', 'CRM', 'TMO',
    'INTC', 'KO', 'PEP', 'CSCO', 'XOM', 'CVX', 'MCD', 'COST', 'ACN', 'WMT', 'BA',
    'ORCL', 'SAP', 'TM', 'SNE', 'SIEGY', 'RYAAY', 'LMT', 'GD', 'MMM', 'CAT', 'GE',
    'F', 'GM', 'BABA', 'JD', 'BIDU', 'PINS', 'SNAP', 'SQ', 'TDOC', 'ZM', 'CRWD',
    'OKTA', 'WORK', 'ETSY', 'ROKU', 'PTON', 'DASH', 'MRNA', 'BNTX', 'PFE', 'LLY',
    'MRK', 'ABBV', 'AMGN', 'GILD', 'BIIB', 'REGN', 'ISRG', 'SYK', 'DHR', 'SPG',
    'O', 'EQIX', 'AMT', 'PLD', 'PSA', 'VTR', 'AVB', 'T', 'VZ', 'S', 'DISH', 'ATT',
    'TMUS', 'TMUB', 'TSN', 'K', 'HSY', 'MDLZ', 'KHC', 'GIS', 'SJM', 'CL', 'CLX',
    'EL', 'AEP', 'DUK', 'NEE', 'XEL', 'SO', 'FE', 'SRE', 'WEC', 'PCG', 'EXC',
    'NI', 'BCE', 'CM', 'TD', 'BNS', 'BMO', 'RY', 'CUK', 'HSBC', 'SAN', 'BBVA',
    'ING', 'AZN', 'GSK', 'SNY', 'NVS', 'ROG', 'NOVN', 'DEO', 'UL', 'PZZA', 'BJRI',
    'CAKE', 'DENN', 'NDSN', 'DOV', 'ITW', 'PH', 'FLS', 'AOS', 'WGO', 'THO', 'BWA',
    'LEVI', 'PVH', 'RL', 'KORS', 'CPRI', 'TGT', 'KR', 'AAL', 'UAL', 'DAL', 'LUV',
    'ALK', 'SAVE', 'JBLU', 'MGM', 'LVS', 'WYNN', 'MLCO', 'MAR', 'HLT', 'IHG',
    'CCL', 'RCL', 'NCLH', 'PENN', 'DKNG', 'FLTR', 'WBD', 'FOXA', 'PARA', 'DISCA',
    'VIAC', 'NWSA', 'TRIP', 'EXPE', 'BKNG', 'ABNB', 'EBAY', 'ETSY', 'WIX', 'SHOP',
    'ZM', 'DOCU', 'DDOG', 'SNOW', 'MDB', 'TEAM', 'ADSK', 'ANSS', 'CDNS', 'SNPS'
    # Adding a few hundred mock tickers to reach 500+ for the 'universe' request
] + [f'MOCK{i:03d}' for i in range(1, 400)])))


# --- Helper Functions (Memoized for Efficiency) ---

def format_number(n, is_currency=False, decimals=2):
    """Formats large numbers for display."""
    if n is None or n == 'N/A' or not isinstance(n, (int, float)):
        return 'N/A'
    
    if is_currency:
        if abs(n) >= 1e12:
            return f"${n/1e12:.{decimals}f}T"
        elif abs(n) >= 1e9:
            return f"${n/1e9:.{decimals}f}B"
        elif abs(n) >= 1e6:
            return f"${n/1e6:.{decimals}f}M"
        else:
            return f"${n:,.{decimals}f}"
    else:
        return f"{n:,.{decimals}f}"


@st.cache_data(show_spinner="Fetching stock price data...")
def fetch_price_data(tickers, start_date, end_date):
    """Downloads historical adjusted closing prices."""
    if not tickers:
        return pd.DataFrame()
    
    real_tickers = [t for t in tickers if not t.startswith('MOCK')]
    if not real_tickers:
        return pd.DataFrame()

    try:
        # Get 'Adj Close' for all tickers
        data = yf.download(real_tickers, start=start_date, end=end_date, progress=False)['Adj Close']
        
        # Ensure result is a DataFrame even for a single ticker
        if isinstance(data, pd.Series):
            data = data.to_frame(name=real_tickers[0])
            
        return data.dropna()
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner="Fetching fundamental data...")
def fetch_fundamentals(ticker):
    """Fetches key fundamental information and price history for a single ticker."""
    if ticker.startswith('MOCK'):
        # Mock data for demonstration
        return {
            'symbol': ticker,
            'longName': f'Mock Tech Co. ({ticker})',
            'sector': np.random.choice(['Technology', 'Healthcare', 'Financial Services', 'Energy']),
            'marketCap': np.random.randint(500000000, 50000000000),
            'trailingPE': np.random.uniform(15, 60),
            'dividendYield': np.random.uniform(0, 0.03),
            'longBusinessSummary': f'This is a synthetic description for {ticker}. It specializes in disruptive solutions within the {np.random.choice(["cloud computing", "AI integration", "biotech development"])} sector. Financials are simulated.',
        }, pd.DataFrame({'Close': [100 + i for i in range(252)]}) # Mock prices
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        required_keys = ['longName', 'sector', 'marketCap', 'trailingPE', 'dividendYield', 'longBusinessSummary', 'industry']
        fundamentals = {k: info.get(k, 'N/A') for k in required_keys}
        fundamentals['symbol'] = ticker
        
        # Fetch 1 year of price data for the chart
        hist_prices = stock.history(period="1y")['Close']
        
        return fundamentals, hist_prices
    except Exception:
        return None, None


def calculate_portfolio_metrics(weights, log_returns, annual_trading_days=252):
    """Calculates annualized return, volatility, and Sharpe Ratio."""
    
    # 1. Annualized Return (Expected Return)
    # Sum of (weight * mean daily return) * trading days
    portfolio_return = np.sum(log_returns.mean() * weights) * annual_trading_days
    
    # 2. Annualized Volatility (Standard Deviation)
    # Volatility = sqrt(w^T * Cov * w) * sqrt(trading days)
    cov_matrix = log_returns.cov() * annual_trading_days
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # 3. Sharpe Ratio (Assuming a 2% risk-free rate, which is common for short-term T-bills)
    RISK_FREE_RATE = 0.02 
    sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility
    
    return portfolio_return, portfolio_volatility, sharpe_ratio


@st.cache_data(show_spinner="Running Monte Carlo Simulation...")
def monte_carlo_optimization(stock_data, num_portfolios=10000):
    """
    Performs Monte Carlo simulation to estimate the Efficient Frontier.
    """
    if stock_data.empty:
        return pd.DataFrame(), {}, {}

    # Calculate Daily Log Returns
    log_returns = np.log(stock_data / stock_data.shift(1)).dropna()
    num_assets = len(stock_data.columns)

    # Initialize storage arrays
    all_weights = np.zeros((num_portfolios, num_assets))
    ret_arr = np.zeros(num_portfolios)
    vol_arr = np.zeros(num_portfolios)
    sharpe_arr = np.zeros(num_portfolios)
    
    # Run Simulation
    for i in range(num_portfolios):
        # Generate random weights that sum to 1
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        all_weights[i, :] = weights
        
        # Calculate portfolio metrics
        ret, vol, sharpe = calculate_portfolio_metrics(weights, log_returns)
        
        ret_arr[i] = ret
        vol_arr[i] = vol
        sharpe_arr[i] = sharpe

    # Compile results
    results_data = {
        'Return': ret_arr,
        'Volatility': vol_arr,
        'Sharpe Ratio': sharpe_arr
    }
    
    for i, asset in enumerate(stock_data.columns):
        results_data[asset + ' Weight'] = all_weights[:, i]

    results_frame = pd.DataFrame(results_data)

    # Find Optimal Portfolios
    max_sharpe_idx = results_frame['Sharpe Ratio'].idxmax()
    min_vol_idx = results_frame['Volatility'].idxmin()

    def extract_optimal_portfolio(idx):
        """Helper to format the optimal portfolio results."""
        weights_series = results_frame.iloc[idx][[col for col in results_frame.columns if 'Weight' in col]]
        metrics = {
            'Return': results_frame.loc[idx, 'Return'],
            'Volatility': results_frame.loc[idx, 'Volatility'],
            'Sharpe Ratio': results_frame.loc[idx, 'Sharpe Ratio'],
            'Weights': pd.DataFrame({
                'Asset': weights_series.index.str.replace(' Weight', ''),
                'Allocation (%)': (weights_series.values * 100).round(2)
            }).set_index('Asset')
        }
        return metrics

    max_sharpe_portfolio = extract_optimal_portfolio(max_sharpe_idx)
    min_vol_portfolio = extract_optimal_portfolio(min_vol_idx)

    return results_frame, max_sharpe_portfolio, min_vol_portfolio


# --- Streamlit Application Pages ---

def page_fundamental_analysis():
    """Displays the Fundamental Analysis page."""
    st.title("🔬 Fundamental Stock Analysis")
    st.write("View fundamental metrics and price charts for any stock in the universe.")
    
    col1, col2 = st.columns([3, 1])
    
    # Use a large subset for the search dropdown
    search_options = TICKER_UNIVERSE
    
    selected_ticker = col1.selectbox(
        "Select or type a ticker:", 
        options=search_options,
        index=search_options.index('AAPL') if 'AAPL' in search_options else 0,
        key=f'{appId}_ticker_select'
    )
    
    if selected_ticker and col2.button("Analyze Stock"):
        with st.spinner(f"Fetching data for {selected_ticker}..."):
            fundamentals, prices = fetch_fundamentals(selected_ticker)
            
            if fundamentals and (selected_ticker.startswith('MOCK') or (prices is not None and not prices.empty)):
                
                st.subheader(f"{fundamentals.get('longName', selected_ticker)} ({selected_ticker})")
                st.markdown(f"**Sector:** {fundamentals.get('sector', 'N/A')} | **Industry:** {fundamentals.get('industry', 'N/A')}")
                
                if selected_ticker.startswith('MOCK'):
                    st.warning("Note: This is simulated data for a mock ticker.")
                
                st.markdown("---")
                
                # Key Metrics Display
                colA, colB, colC, colD = st.columns(4)
                
                colA.metric("Market Cap", format_number(fundamentals.get('marketCap'), is_currency=True, decimals=2))
                colB.metric("P/E Ratio (TTM)", format_number(fundamentals.get('trailingPE')))
                
                div_yield = fundamentals.get('dividendYield')
                div_str = f"{div_yield * 100:.2f}%" if isinstance(div_yield, (int, float)) and div_yield > 0 else '0.00%'
                colC.metric("Div Yield", div_str)
                
                colD.metric("Symbol", selected_ticker)

                # Historical Price Chart
                st.subheader("Historical Price (Last Year)")
                if prices is not None and not prices.empty:
                    fig = px.line(
                        prices.to_frame(name='Close'), 
                        y='Close', 
                        title=f'{selected_ticker} Adjusted Closing Price',
                        labels={'Close': 'Price (USD)', 'index': 'Date'}
                    )
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Price chart is not available for this period or ticker.")
                
                # Business Summary
                with st.expander("Company Business Summary"):
                    st.write(fundamentals.get('longBusinessSummary', 'No business summary available.'))
                    
            else:
                st.error(f"Could not retrieve valid data for ticker '{selected_ticker}'. Please verify the symbol.")


def page_portfolio_optimizer():
    """Displays the Portfolio Optimization page (Markowitz Efficient Frontier)."""
    st.title("📈 Portfolio Optimizer (Mean-Variance)")
    st.write("Applies Modern Portfolio Theory (MPT) using Monte Carlo simulation to optimize asset allocation based on risk and return.")

    # --- Setup and Input ---
    
    DEFAULT_ASSETS = ['AAPL', 'MSFT', 'JPM', 'XOM', 'GLD']
    
    selected_assets = st.multiselect(
        "Select Assets for Optimization (Min 2):",
        options=[t for t in TICKER_UNIVERSE if not t.startswith('MOCK')],
        default=DEFAULT_ASSETS,
        key=f'{appId}_assets_select'
    )
    
    col1, col2 = st.columns(2)
    with col1:
        years = st.slider("Historical Data Period (Years):", 1, 10, 5, key=f'{appId}_years_slider')
    with col2:
        num_runs = st.slider("Monte Carlo Simulations:", 1000, 20000, 10000, 1000, key=f'{appId}_runs_slider')

    start_date = (datetime.now() - timedelta(days=int(years * 365.25))).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    st.info(f"Optimization will use historical data from **{start_date}** to **{end_date}**.")
    
    st.markdown("---")

    # --- Optimization Logic ---
    if st.button("Run Portfolio Optimization", use_container_width=True):
        if len(selected_assets) < 2:
            st.warning("Please select at least two assets to perform portfolio optimization.")
            return

        # 1. Fetch data
        stock_data = fetch_price_data(selected_assets, start_date, end_date)
        
        if stock_data.empty or len(stock_data.columns) < 2:
            st.error("Could not retrieve enough valid historical data for all selected assets. Check tickers/period.")
            return

        # 2. Run optimization
        results_frame, max_sharpe, min_vol = monte_carlo_optimization(stock_data, num_runs)

        # 3. Output and Visualization
        
        st.subheader("Efficient Frontier Analysis")
        st.success(f"Simulation Complete: Analyzed {num_runs} portfolios over {years} years.")
        
        # Efficient Frontier Plot 
        fig = px.scatter(
            results_frame,
            x='Volatility',
            y='Return',
            color='Sharpe Ratio',
            color_continuous_scale=px.colors.sequential.Viridis,
            title='Markowitz Efficient Frontier (Monte Carlo Simulation)',
            labels={'Volatility': 'Annualized Volatility (Risk)', 'Return': 'Annualized Return'}
        )
        
        # Add Optimal Portfolio markers
        fig.add_trace(go.Scatter(
            x=[max_sharpe['Volatility']],
            y=[max_sharpe['Return']],
            mode='markers',
            marker=dict(color='red', size=15, symbol='star', line=dict(width=1, color='black')),
            name=f'Max Sharpe Ratio ({max_sharpe["Sharpe Ratio"]:.2f})'
        ))

        fig.add_trace(go.Scatter(
            x=[min_vol['Volatility']],
            y=[min_vol['Return']],
            mode='markers',
            marker=dict(color='blue', size=15, symbol='circle', line=dict(width=1, color='black')),
            name=f'Min Volatility ({min_vol["Volatility"]:.2f})'
        ))

        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)


        # 4. Optimal Portfolio Results
        st.subheader("Optimal Portfolio Allocations")
        st.markdown("The following allocations represent the historical optimal trade-offs between risk and return, based on the selected data period.")
        
        col_max, col_min = st.columns(2)
        
        # Max Sharpe Portfolio
        with col_max:
            st.markdown(f"**🔴 Max Sharpe Ratio Portfolio**")
            st.metric("Sharpe Ratio", f"{max_sharpe['Sharpe Ratio']:.2f}")
            st.metric("Expected Return (Annual)", f"{max_sharpe['Return'] * 100:.2f}%")
            st.metric("Expected Volatility (Annual)", f"{max_sharpe['Volatility'] * 100:.2f}%")
            
            st.dataframe(max_sharpe['Weights'], use_container_width=True)


        # Min Volatility Portfolio
        with col_min:
            st.markdown(f"**🔵 Minimum Volatility Portfolio**")
            st.metric("Sharpe Ratio", f"{min_vol['Sharpe Ratio']:.2f}")
            st.metric("Expected Return (Annual)", f"{min_vol['Return'] * 100:.2f}%")
            st.metric("Expected Volatility (Annual)", f"{min_vol['Volatility'] * 100:.2f}%")
            
            st.dataframe(min_vol['Weights'], use_container_width=True)
        

# --- Main App Execution ---
def main():
    st.set_page_config(
        page_title="Financial Portfolio Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Global Style: Custom CSS for aesthetics
    st.markdown("""
        <style>
        /* General page layout improvements */
        .css-1d391kg { padding-top: 35px; } 
        
        /* Button styling for better UX */
        .stButton>button {
            background-color: #007BFF;
            color: white;
            border-radius: 8px;
            font-weight: bold;
            transition: all 0.2s;
            border: 1px solid #007BFF;
        }
        .stButton>button:hover {
            background-color: #0056b3;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        /* Metric card styling */
        [data-testid="stMetric"] {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.05);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.title("Financial Analysis Tools")
    st.sidebar.markdown(f"**App ID:** `{appId}`")
    
    menu_selection = st.sidebar.radio(
        "Choose a Tool:",
        ("Portfolio Optimizer", "Fundamental Analysis"),
        key=f'{appId}_menu'
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Stock Universe Size:** {len(TICKER_UNIVERSE)} Symbols")
    st.sidebar.caption("Includes large, mid, and mock small-cap tickers.")
    st.sidebar.caption("Data is powered by Yahoo Finance (`yfinance`).")

    # Render selected page
    if menu_selection == "Fundamental Analysis":
        page_fundamental_analysis()
    elif menu_selection == "Portfolio Optimizer":
        page_portfolio_optimizer()
        
if __name__ == '__main__':
    main()
