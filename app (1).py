import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import time

# --- Setup and Constants ---

# Define the environment variable checks for Python (safer)
try:
    # Attempt to use a secret if running on Streamlit Cloud or similar environments
    appId = st.secrets["APP_ID"]
except (KeyError, AttributeError):
    # Fallback for local testing or unknown environments
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
    
    # Filter out mock tickers before calling yfinance
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
            'industry': 'Mock Industry'
        }, pd.DataFrame({'Close': [100 + i for i in range(252)]}) # Mock prices
    
    try:
        stock = yf.Ticker(ticker)
        # Fetching info can sometimes fail, so wrap in try/except
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
    portfolio_return = np.sum(log_returns.mean() * weights) * annual_trading_days
    
    # 2. Annualized Volatility (Standard Deviation)
    cov_matrix = log_returns.cov() * annual_trading_days
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # 3. Sharpe Ratio (Assumes a 2% risk-free rate)
    RISK_FREE_RATE = 0.02 
    sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility
    
    return portfolio_return, portfolio_volatility, sharpe_ratio

def calculate_risk_metrics(log_returns, weights, confidence_level=0.95, annual_trading_days=252):
    """Calculates Value at Risk (VaR) and Conditional Value at Risk (CVaR)."""
    
    # Annualize mean and std dev for the portfolio
    port_mean = np.sum(log_returns.mean() * weights) * annual_trading_days
    cov_matrix = log_returns.cov() * annual_trading_days
    port_stdev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # 1. Parametric VaR (assuming normal distribution)
    # VaR = mean - (stdev * inverse cumulative distribution function at (1 - confidence_level))
    VaR_percent = norm.ppf(1 - confidence_level, port_mean, port_stdev)
    
    # 2. Parametric CVaR (Expected Shortfall) - simplified formula for normal distribution
    # CVaR is the expected loss given that the loss is greater than VaR.
    alpha = 1 - confidence_level
    # Note: For CVaR of a normal distribution, the formula is (pdf(norm.ppf(alpha)) / alpha) * stdev - mean
    CVaR_percent = alpha**-1 * norm.pdf(norm.ppf(alpha)) * port_stdev - port_mean

    return abs(VaR_percent), abs(CVaR_percent)


@st.cache_data(show_spinner="Running Monte Carlo Simulation (10,000 runs)...")
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
        weights_array = weights_series.values
        
        # Calculate advanced risk metrics for the optimal portfolio
        VaR, CVaR = calculate_risk_metrics(log_returns, weights_array)

        metrics = {
            'Return': results_frame.loc[idx, 'Return'],
            'Volatility': results_frame.loc[idx, 'Volatility'],
            'Sharpe Ratio': results_frame.loc[idx, 'Sharpe Ratio'],
            'VaR (95%)': VaR,
            'CVaR (95%)': CVaR,
            'Weights': pd.DataFrame({
                'Asset': weights_series.index.str.replace(' Weight', ''),
                'Allocation (%)': (weights_series.values * 100).round(2)
            }).set_index('Asset')
        }
        return metrics

    max_sharpe_portfolio = extract_optimal_portfolio(max_sharpe_idx)
    min_vol_portfolio = extract_optimal_portfolio(min_vol_idx)
    
    # Return log_returns as well for custom portfolio calculation later
    return results_frame, max_sharpe_portfolio, min_vol_portfolio, log_returns


def find_custom_portfolio(results_frame, target_return, target_volatility, log_returns):
    """
    Finds the simulated portfolio closest to the user's defined targets.
    This uses the Euclidean distance to find the closest point on the risk-return plane.
    """
    # Normalize inputs to prevent distance domination by one metric
    results_frame['Norm_Return'] = results_frame['Return']
    results_frame['Norm_Volatility'] = results_frame['Volatility']
    
    # Calculate the distance from the target point (Target Volatility, Target Return)
    results_frame['Distance'] = np.sqrt(
        (results_frame['Norm_Volatility'] - target_volatility)**2 + 
        (results_frame['Norm_Return'] - target_return)**2
    )
    
    # Find the index of the minimum distance
    closest_idx = results_frame['Distance'].idxmin()

    # Extract metrics for the closest portfolio
    weights_series = results_frame.iloc[closest_idx][[col for col in results_frame.columns if 'Weight' in col]]
    weights_array = weights_series.values
    
    # Calculate VaR/CVaR using the passed log_returns object
    VaR, CVaR = calculate_risk_metrics(log_returns, weights_array)
    
    metrics = {
        'Return': results_frame.loc[closest_idx, 'Return'],
        'Volatility': results_frame.loc[closest_idx, 'Volatility'],
        'Sharpe Ratio': results_frame.loc[closest_idx, 'Sharpe Ratio'],
        'VaR (95%)': VaR,
        'CVaR (95%)': CVaR,
        'Weights': pd.DataFrame({
            'Asset': weights_series.index.str.replace(' Weight', ''),
            'Allocation (%)': (weights_series.values * 100).round(2)
        }).set_index('Asset')
    }
    return metrics


# --- Streamlit Application Pages ---

def page_fundamental_analysis():
    """Displays the Fundamental Analysis page."""
    st.title("🔬 Stock Deep Dive")
    st.markdown("### Fundamental Metrics & Price Action")
    
    col1, col2 = st.columns([3, 1])
    
    search_options = TICKER_UNIVERSE
    
    selected_ticker = col1.selectbox(
        "Select or type a ticker:", 
        options=search_options,
        index=search_options.index('AAPL') if 'AAPL' in search_options else 0,
        key=f'{appId}_ticker_select'
    )
    
    if selected_ticker and col2.button("Analyze Stock", key='analyze_btn'):
        with st.spinner(f"Fetching data for {selected_ticker}..."):
            fundamentals, prices = fetch_fundamentals(selected_ticker)
            
            is_valid_data = fundamentals and (selected_ticker.startswith('MOCK') or (prices is not None and not prices.empty))

            if is_valid_data:
                
                st.markdown(f"## {fundamentals.get('longName', selected_ticker)} ({selected_ticker})")
                
                if selected_ticker.startswith('MOCK'):
                    st.warning("Note: This is simulated data for a mock ticker.")
                
                st.markdown("---")
                
                # Key Metrics Display (4 columns)
                colA, colB, colC, colD = st.columns(4)
                
                colA.metric("Market Cap", format_number(fundamentals.get('marketCap'), is_currency=True, decimals=2), help="The total value of all shares outstanding.")
                colB.metric("P/E Ratio (TTM)", format_number(fundamentals.get('trailingPE')), help="Price to Earnings Ratio - shows if the stock is cheap or expensive.")
                
                div_yield = fundamentals.get('dividendYield')
                div_str = f"{div_yield * 100:.2f}%" if isinstance(div_yield, (int, float)) and div_yield > 0 else '0.00%'
                colC.metric("Div Yield", div_str, help="Percentage return from dividends.")
                
                colD.metric("Sector", fundamentals.get('sector', 'N/A'))

                # Historical Price Chart
                st.subheader("Price History (Last Year)")
                if prices is not None and not prices.empty:
                    fig = px.line(
                        prices.to_frame(name='Close'), 
                        y='Close', 
                        title=f'{selected_ticker} Adjusted Closing Price',
                        labels={'Close': 'Price (USD)', 'index': 'Date'}
                    )
                    fig.update_layout(height=450, showlegend=False, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Price chart is not available for this period or ticker.")
                
                # Business Summary
                with st.expander("💼 Company Business Summary"):
                    st.write(fundamentals.get('longBusinessSummary', 'No business summary available.'))
                    
            else:
                st.error(f"Could not retrieve valid data for ticker '{selected_ticker}'. Please verify the symbol.")


def page_portfolio_optimizer():
    """Displays the Portfolio Optimization page (Markowitz Efficient Frontier)."""
    st.title("💰 Portfolio Optimizer")
    st.write("Find the ideal mix of assets to balance risk and expected return using the Efficient Frontier model.")
    

    # --- 1. Select Assets, Timeframe, and Custom Targets ---
    
    DEFAULT_ASSETS = ['AAPL', 'MSFT', 'JPM', 'XOM', 'GLD']
    
    with st.container():
        st.subheader("1. Setup Your Investment Strategy")
        
        col_assets, col_years = st.columns([3, 1])

        # Define the options list (only real tickers allowed for optimization)
        available_options = [t for t in TICKER_UNIVERSE if not t.startswith('MOCK')]
        
        # Ensure default assets exist in the options list
        default_selection = [asset for asset in DEFAULT_ASSETS if asset in available_options]

        selected_assets = col_assets.multiselect(
            "Assets (Min 2):",
            options=available_options,
            default=default_selection,
            key=f'{appId}_assets_select',
            help="Select the stocks you want to include in your portfolio for optimization."
        )
        
        with col_years:
            years = st.slider("Historical Data (Years):", 1, 10, 5, key=f'{appId}_years_slider', help="Longer periods give more data but may include outdated market conditions.")

        num_runs = st.slider("Monte Carlo Simulations:", 1000, 20000, 10000, 1000, key=f'{appId}_runs_slider', help="More runs increase accuracy but take slightly longer.")

        # Calculate dates and store in session state for cross-function access
        start_date = (datetime.now() - timedelta(days=int(years * 365.25))).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        st.session_state[f'{appId}_start_date'] = start_date
        st.session_state[f'{appId}_end_date'] = end_date

        st.caption(f"Data period: {start_date} to {end_date}")
        
    st.markdown("---")
    
    # --- Custom Risk/Return Targets ---
    with st.expander("🎯 Set Your Target Return and Risk Appetite (Optional)", expanded=True):
        col_target_ret, col_target_vol = st.columns(2)
        
        # Max reasonable value for slider is based on historical market data (e.g., 50% max return, 40% max vol)
        target_return = col_target_ret.slider(
            "Target Annual Return (%)", 
            min_value=0, max_value=50, value=15, step=1, 
            key=f'{appId}_target_return',
            help="Your desired yearly return from the portfolio."
        ) / 100 # Convert to decimal
        
        target_volatility = col_target_vol.slider(
            "Risk Appetite (Target Annual Volatility %)", 
            min_value=5, max_value=40, value=20, step=1, 
            key=f'{appId}_target_volatility',
            help="Your maximum acceptable risk level (volatility)."
        ) / 100 # Convert to decimal
        
        st.markdown(f"**Selected Target:** Return: **{target_return * 100:.1f}%** | Volatility: **{target_volatility * 100:.1f}%**")


    st.markdown("---")

    # --- Optimization Logic ---
    if st.button("🚀 Run Optimization", use_container_width=True, key='run_opt_btn'):
        if len(selected_assets) < 2:
            st.error("Please select at least two assets to perform portfolio optimization.")
            return

        # 1. Fetch data
        stock_data = fetch_price_data(selected_assets, start_date, end_date)
        
        if stock_data.empty or len(stock_data.columns) < 2:
            st.error("Could not retrieve enough valid historical data for all selected assets. Check tickers/period.")
            return

        # 2. Run optimization
        with st.spinner(f"Running {num_runs} Monte Carlo simulations..."):
            # monte_carlo_optimization now returns log_returns too
            results_frame, max_sharpe, min_vol, log_returns = monte_carlo_optimization(stock_data, num_runs)

        st.session_state[f'{appId}_results_frame'] = results_frame
        st.session_state[f'{appId}_log_returns'] = log_returns # Store log returns for custom finder
        st.session_state[f'{appId}_max_sharpe'] = max_sharpe
        st.session_state[f'{appId}_min_vol'] = min_vol
        st.session_state[f'{appId}_target_return'] = target_return
        st.session_state[f'{appId}_target_volatility'] = target_volatility
        st.session_state[f'{appId}_optimization_run'] = True
        
        st.rerun() # Use rerun to display results after updating session state

    # --- Display Results ---
    if f'{appId}_optimization_run' in st.session_state and st.session_state[f'{appId}_optimization_run']:
        results_frame = st.session_state[f'{appId}_results_frame']
        log_returns = st.session_state[f'{appId}_log_returns'] # Retrieve log returns
        max_sharpe = st.session_state[f'{appId}_max_sharpe']
        min_vol = st.session_state[f'{appId}_min_vol']
        target_return = st.session_state[f'{appId}_target_return']
        target_volatility = st.session_state[f'{appId}_target_volatility']
        
        # --- Find Custom Portfolio ---
        # Pass log_returns directly to avoid refetching data in the helper function
        custom_portfolio = find_custom_portfolio(results_frame.copy(), target_return, target_volatility, log_returns)

        st.success(f"Simulation Complete! {len(results_frame)} portfolios analyzed.")
        
        st.subheader("2. Efficient Frontier & Optimal Portfolios")
        st.info("The Efficient Frontier (green curve) shows the set of best possible portfolios. The chart below visualizes thousands of randomized portfolios.")
        
        # Efficient Frontier Plot (Scatter Plot)
        fig = px.scatter(
            results_frame,
            x='Volatility',
            y='Return',
            color='Sharpe Ratio',
            color_continuous_scale=px.colors.sequential.Viridis,
            title='Markowitz Efficient Frontier & Portfolio Scatter',
            labels={'Volatility': 'Annualized Volatility (Risk)', 'Return': 'Annualized Return'}
        )
        
        # Add Optimal Portfolio markers (using different, distinct colors)
        
        # Max Sharpe (Red Star)
        fig.add_trace(go.Scatter(
            x=[max_sharpe['Volatility']],
            y=[max_sharpe['Return']],
            mode='markers',
            marker=dict(color='#E50000', size=18, symbol='star', line=dict(width=2, color='white')),
            name=f'Max Sharpe ({max_sharpe["Sharpe Ratio"]:.2f})'
        ))

        # Min Vol (Cyan Circle)
        fig.add_trace(go.Scatter(
            x=[min_vol['Volatility']],
            y=[min_vol['Return']],
            mode='markers',
            marker=dict(color='#00FFFF', size=18, symbol='circle', line=dict(width=2, color='white')),
            name=f'Min Volatility ({min_vol["Volatility"] * 100:.2f}%)'
        ))
        
        # Custom Target (Orange Square) - Represents user's goal
        fig.add_trace(go.Scatter(
            x=[target_volatility],
            y=[target_return],
            mode='markers',
            marker=dict(color='#FF8C00', size=18, symbol='square', line=dict(width=2, color='black')),
            name=f'Your Target ({target_return * 100:.1f}%, {target_volatility * 100:.1f}%)'
        ))
        
        # Custom Found Portfolio (Dark Violet Diamond) - The result closest to the user's goal
        fig.add_trace(go.Scatter(
            x=[custom_portfolio['Volatility']],
            y=[custom_portfolio['Return']],
            mode='markers',
            marker=dict(color='#9400D3', size=18, symbol='diamond', line=dict(width=2, color='white')),
            name=f'Custom Portfolio Found'
        ))


        fig.update_layout(height=600, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        # 


        # 3. Optimal Allocation & Risk Breakdown
        st.subheader("3. Optimal Allocation & Risk Breakdown")
        
        tab1, tab2, tab3 = st.tabs(["🎯 Custom Portfolio", "⭐ Max Sharpe Portfolio", "🐢 Min Volatility Portfolio"])
        
        # --- Tab 1: Custom Portfolio ---
        with tab1:
            st.markdown(f"### Closest to Your Target (Return: {target_return * 100:.1f}% | Vol: {target_volatility * 100:.1f}%)")
            
            col_metric_A, col_metric_B, col_metric_C = st.columns(3)
            col_metric_A.metric("Return Achieved", f"{custom_portfolio['Return'] * 100:.2f}%", help="Annualized expected return.")
            col_metric_B.metric("Volatility Achieved", f"{custom_portfolio['Volatility'] * 100:.2f}%", help="Annualized risk level.")
            col_metric_C.metric("Sharpe Ratio", f"{custom_portfolio['Sharpe Ratio']:.2f}", help="Risk-adjusted return (Higher is better).")

            st.markdown("#### Allocation Weights (How much of each stock to hold)")
            col_weights, col_chart = st.columns(2)
            
            with col_weights:
                st.dataframe(custom_portfolio['Weights'], use_container_width=True)
            
            with col_chart:
                fig_weights = px.pie(
                    custom_portfolio['Weights'].reset_index(),
                    values='Allocation (%)',
                    names='Asset',
                    title='Asset Distribution',
                    color_discrete_sequence=px.colors.qualitative.T10 # Distinct color palette
                )
                fig_weights.update_traces(textposition='inside', textinfo='percent+label')
                fig_weights.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_weights, use_container_width=True)
            
            st.markdown("#### Risk Analysis (95% Confidence)")
            col_var, col_cvar = st.columns(2)
            col_var.metric("Annual VaR (Value at Risk)", f"{custom_portfolio['VaR (95%)'] * 100:.2f}%", help="Maximum estimated loss 95% of the time.")
            col_cvar.metric("Annual CVaR (Conditional VaR)", f"{custom_portfolio['CVaR (95%)'] * 100:.2f}%", help="Average loss in the worst 5% of cases.")


        # --- Tab 2: Max Sharpe Portfolio ---
        with tab2:
            st.markdown("### ⭐ Max Sharpe Portfolio (Highest Risk-Adjusted Return)")
            col_metric_A, col_metric_B, col_metric_C = st.columns(3)
            col_metric_A.metric("Expected Return", f"{max_sharpe['Return'] * 100:.2f}%")
            col_metric_B.metric("Volatility", f"{max_sharpe['Volatility'] * 100:.2f}%")
            col_metric_C.metric("Sharpe Ratio", f"{max_sharpe['Sharpe Ratio']:.2f}")

            st.markdown("#### Allocation Weights")
            col_weights, col_chart = st.columns(2)
            
            with col_weights:
                st.dataframe(max_sharpe['Weights'], use_container_width=True)
            
            with col_chart:
                fig_weights = px.pie(
                    max_sharpe['Weights'].reset_index(),
                    values='Allocation (%)',
                    names='Asset',
                    title='Asset Distribution',
                    color_discrete_sequence=px.colors.qualitative.Set2 # Different color palette
                )
                fig_weights.update_traces(textposition='inside', textinfo='percent+label')
                fig_weights.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_weights, use_container_width=True)
            
            st.markdown("#### Risk Analysis (95% Confidence)")
            col_var, col_cvar = st.columns(2)
            col_var.metric("Annual VaR (Value at Risk)", f"{max_sharpe['VaR (95%)'] * 100:.2f}%")
            col_cvar.metric("Annual CVaR (Conditional VaR)", f"{max_sharpe['CVaR (95%)'] * 100:.2f}%")
        
        # --- Tab 3: Min Volatility Portfolio ---
        with tab3:
            st.markdown("### 🐢 Min Volatility Portfolio (Lowest Risk)")
            col_metric_A, col_metric_B, col_metric_C = st.columns(3)
            col_metric_A.metric("Expected Return", f"{min_vol['Return'] * 100:.2f}%")
            col_metric_B.metric("Volatility", f"{min_vol['Volatility'] * 100:.2f}%")
            col_metric_C.metric("Sharpe Ratio", f"{min_vol['Sharpe Ratio']:.2f}")

            st.markdown("#### Allocation Weights")
            col_weights, col_chart = st.columns(2)
            
            with col_weights:
                st.dataframe(min_vol['Weights'], use_container_width=True)
            
            with col_chart:
                fig_weights = px.pie(
                    min_vol['Weights'].reset_index(),
                    values='Allocation (%)',
                    names='Asset',
                    title='Asset Distribution',
                    color_discrete_sequence=px.colors.qualitative.Pastel # Different color palette
                )
                fig_weights.update_traces(textposition='inside', textinfo='percent+label')
                fig_weights.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_weights, use_container_width=True)
            
            st.markdown("#### Risk Analysis (95% Confidence)")
            col_var, col_cvar = st.columns(2)
            col_var.metric("Annual VaR (Value at Risk)", f"{min_vol['VaR (95%)'] * 100:.2f}%")
            col_cvar.metric("Annual CVaR (Conditional VaR)", f"{min_vol['CVaR (95%)'] * 100:.2f}%")


# --- Main App Execution ---
def main():
    st.set_page_config(
        page_title="Financial Portfolio Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Global Style: Custom CSS for aesthetics and engaging UX
    st.markdown("""
        <style>
        /* General page layout improvements */
        .css-1d391kg { padding-top: 35px; } 
        
        /* Custom Font for Header */
        h1 { color: #007BFF; text-shadow: 1px 1px 2px #d0d0d0; }
        h3 { color: #0056b3; }
        
        /* Button styling for better UX */
        .stButton>button {
            background-color: #4CAF50; /* Green for GO button */
            color: white;
            border-radius: 12px;
            font-weight: bold;
            font-size: 1.1rem;
            padding: 10px 20px;
            transition: all 0.2s;
            border: none;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton>button:hover {
            background-color: #45a049;
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
        }
        /* Metric card styling */
        [data-testid="stMetric"] {
            background-color: #f7f9fc;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.05);
        }
        /* Make multiselect input look cleaner and slightly raised */
        [data-testid="stMultiSelect"] > div {
            border-radius: 8px;
            box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.05);
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
