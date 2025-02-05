from synth.miner.price_simulation import (
    simulate_crypto_price_paths,
    get_asset_price,
)
from synth.utils.helpers import (
    convert_prices_to_time_format,
)

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

class CryptoForecaster:
    def __init__(self):
        self.model_name = "ElKulako/cryptobert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    def get_historical_data(self, symbol="BTC-USD", period="60d", interval="5m"):
        """Fetch historical price data"""
        try:
            data = yf.download(symbol, period=period, interval=interval)
            return data['Close']
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None

    def get_market_sentiment(self, price_data):
        """Get market sentiment using CryptoBERT"""
        with torch.no_grad():
            returns = price_data.pct_change().dropna()
            sentiment_text = f"Price {'increased' if returns.iloc[-1] > 0 else 'decreased'} by {abs(returns.iloc[-1]*100):.2f}%"
            
            inputs = self.tokenizer(sentiment_text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            sentiment_score = torch.sigmoid(outputs.logits).cpu().numpy()[0][0]
            return sentiment_score

    def generate_price_paths(self, current_price, start_time, time_increment, time_length, num_simulations):
        """
        Generate price paths following validator requirements:
        - Correct time format
        - Proper increments
        - Required number of simulations
        """
        steps = int(time_length / time_increment) + 1
        historical_data = self.get_historical_data()
        returns = np.log(historical_data / historical_data.shift(1)).dropna()
        
        volatility = returns.std() * np.sqrt(time_increment / 3600)
        drift = returns.mean()
        sentiment_score = self.get_market_sentiment(historical_data)
        
        paths = []
        
        for sim in range(num_simulations):
            path = []
            current_time = datetime.fromisoformat(start_time)
            current_sim_price = current_price
            
            for step in range(steps):
                path.append({
                    "time": current_time.isoformat(),
                    "price": float(current_sim_price)
                })
                
                if step < steps - 1:
                    random_shock = np.random.normal(0, volatility)
                    sentiment_adjustment = (sentiment_score - 0.5) * volatility
                    
                    price_change = (drift - 0.5 * volatility**2) * (time_increment/3600) + \
                                 random_shock + sentiment_adjustment
                    
                    current_sim_price *= np.exp(price_change)
                    current_time += timedelta(seconds=time_increment)
            
            paths.append(path)
        
        return paths

def generate_simulations(
    asset="BTC",
    start_time=None,
    time_increment=300,
    time_length=86400,
    num_simulations=100,
):
    """
    Generate simulations following validator requirements
    """
    if start_time is None:
        raise ValueError("Start time must be provided.")
    
    forecaster = CryptoForecaster()
    current_price = yf.Ticker(f"{asset}-USD").history(period="1m")['Close'].iloc[-1]
    
    predictions = forecaster.generate_price_paths(
        current_price=current_price,
        start_time=start_time,
        time_increment=time_increment,
        time_length=time_length,
        num_simulations=num_simulations
    )

    final_predictions = convert_prices_to_time_format(predictions, start_time, time_increment)
    
    return final_predictions

def _generate_simulations(
    asset="BTC",
    start_time=None,
    time_increment=300,
    time_length=86400,
    num_simulations=1,
):
    """
    Generate simulated price paths.

    Parameters:
        asset (str): The asset to simulate. Default is 'BTC'.
        start_time (str): The start time of the simulation. Defaults to current time.
        time_increment (int): Time increment in seconds.
        time_length (int): Total time length in seconds.
        num_simulations (int): Number of simulation runs.

    Returns:
        numpy.ndarray: Simulated price paths.
    """
    if start_time is None:
        raise ValueError("Start time must be provided.")

    current_price = get_asset_price(asset)
    if current_price is None:
        raise ValueError(f"Failed to fetch current price for asset: {asset}")

    # Standard deviation of the simulated price path
    sigma = 0.01

    simulations = simulate_crypto_price_paths(
        current_price=current_price,
        time_increment=time_increment,
        time_length=time_length,
        num_simulations=num_simulations,
        sigma=sigma,
    )

    predictions = convert_prices_to_time_format(
        simulations.tolist(), start_time, time_increment
    )

    return predictions
