import numpy as np
import requests
from properscoring import crps_ensemble
import json
from scipy.stats import t, norm

def get_asset_price(asset="BTC"):
    """Original implementation remains unchanged"""
    if asset == "BTC":
        btc_price_id = "e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43"
        endpoint = f"https://hermes.pyth.network/api/latest_price_feeds?ids[]={btc_price_id}"
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            data = response.json()
            if not data or len(data) == 0:
                raise ValueError("No price data received")
            price_feed = data[0]
            price = float(price_feed["price"]["price"]) / (10**8)
            return price
        except Exception as e:
            print(f"Error fetching {asset} price: {str(e)}")
            return None
    return None

def get_historical_data(days=30):
    """Enhanced historical data retrieval with additional metrics"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}"
        response = requests.get(url)
        data = json.loads(response.text)
        prices = np.array([price[1] for price in data["prices"]])
        returns = np.log(prices[1:] / prices[:-1])
        
        volatility = np.std(returns) * np.sqrt(252)
        skewness = np.mean(((returns - np.mean(returns))/np.std(returns))**3)
        kurtosis = np.mean(((returns - np.mean(returns))/np.std(returns))**4)
        
        return returns, volatility, skewness, kurtosis
    except:
        return None, None, None, None

def monte_carlo_path(current_price, num_steps, dt, params):
    """Generate single Monte Carlo path with Student's t innovations"""
    prices = np.zeros(num_steps + 1)
    prices[0] = current_price
    
    mu, sigma, df, jump_intensity, jump_size_mean, jump_size_std = params
    
    # Correlated random numbers using Student's t
    random_numbers = t.rvs(df=df, size=num_steps)
    
    jumps = np.random.binomial(1, jump_intensity, size=num_steps) * \
            np.random.normal(jump_size_mean, jump_size_std, size=num_steps)
    
    for i in range(num_steps):
        if i > 0:
            ret = np.log(prices[i] / prices[i-1])
            sigma = np.sqrt(0.9 * sigma**2 + 0.1 * ret**2)
        
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * random_numbers[i]
        
        prices[i + 1] = prices[i] * np.exp(drift + diffusion + jumps[i])
    
    return prices

def simulate_single_price_path(current_price, time_increment, time_length, sigma):
    """Enhanced single path simulation with Monte Carlo and Student's t"""
    num_steps = int(time_length / time_increment)
    dt = time_increment / (24 * 3600)
    
    hist_returns, hist_vol, skewness, kurtosis = get_historical_data()
    
    if hist_returns is not None:
        df = 6 if kurtosis is None else max(3, 6 * (kurtosis - 3)/(kurtosis - 1))
        sigma = 0.9 * sigma + 0.1 * hist_vol if hist_vol is not None else sigma
    else:
        df = 5  # Default value
    
    # Monte Carlo parameters
    params = (
        0,          # mu (risk-neutral drift)
        sigma,      # volatility
        df,         # Student's t degrees of freedom
        0.01,       # jump intensity
        0,          # jump size mean
        0.02        # jump size std
    )
    
    return monte_carlo_path(current_price, num_steps, dt, params)

def simulate_crypto_price_paths(current_price, time_increment, time_length, num_simulations, sigma):
    """Generate multiple correlated Monte Carlo paths"""
    num_steps = int(time_length / time_increment)
    paths = np.zeros((num_simulations, num_steps + 1))
    
    hist_returns, hist_vol, skewness, kurtosis = get_historical_data()
    
    if hist_returns is not None:
        vol_mean = hist_vol if hist_vol is not None else sigma
        vol_std = np.std([np.std(hist_returns[i:i+10]) for i in range(len(hist_returns)-10)])
    else:
        vol_mean = sigma
        vol_std = sigma * 0.1
    
    # Correlated volatilities using Cholesky decomposition
    correlation = 0.7  # Base correlation between paths
    corr_matrix = correlation * np.ones((num_simulations, num_simulations)) + \
                 (1 - correlation) * np.eye(num_simulations)
    chol = np.linalg.cholesky(corr_matrix)
    
    raw_vols = np.random.normal(0, 1, num_simulations)
    volatilities = vol_mean + vol_std * (chol @ raw_vols)
    volatilities = np.maximum(volatilities, vol_mean * 0.5)
    
    for i in range(num_simulations):
        paths[i] = simulate_single_price_path(
            current_price=current_price,
            time_increment=time_increment,
            time_length=time_length,
            sigma=volatilities[i]
        )
    
    return paths
