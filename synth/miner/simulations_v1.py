from synth.miner.price_simulation import (
    simulate_crypto_price_paths,
    get_asset_price,
)
from synth.utils.helpers import (
    convert_prices_to_time_format,
)
import numpy as np

def generate_simulations(
    asset="BTC",
    start_time=None,
    time_increment=300,
    time_length=86400,
    num_simulations=1,
):
    """
    Generate simulated price paths with proper error handling.
    """
    try:
        print(f"DEBUG: generate_simulations called with: asset={asset}, start_time={start_time}, time_increment={time_increment}, time_length={time_length}, num_simulations={num_simulations}")

        if start_time is None:
            raise ValueError("Start time must be provided.")

        current_price = get_asset_price(asset)
        if current_price is None:
            raise ValueError(f"Failed to fetch current price for asset: {asset}")

        # Validate inputs
        if time_increment <= 0:
            raise ValueError("Time increment must be positive")
        if time_length <= 0:
            raise ValueError("Time length must be positive")
        if num_simulations <= 0:
            raise ValueError("Number of simulations must be positive")

        sigma = 0.01
        print(f"DEBUG: Using current_price={current_price} and sigma={sigma}")

        simulations = simulate_crypto_price_paths(
            current_price=current_price,
            time_increment=time_increment,
            time_length=time_length,
            num_simulations=num_simulations,
            sigma=sigma,
        )
        print(f"DEBUG: Simulations generated with shape {simulations.shape}")

        if np.any(np.isnan(simulations)) or np.any(np.isinf(simulations)):
            raise ValueError("Invalid values in simulation results")
        if simulations.size > 0:
            print(f"DEBUG: simulations generatd, converting prices to time format now")
        print(f"DEBUG: Simulations raw data type: {type(simulations)}, shape: {simulations.shape}")
        simulations_list = simulations.tolist()


        # Ensure all elements are lists, not NumPy objects
        if any(isinstance(item, np.ndarray) for item in simulations_list):
            raise ValueError("ERROR: simulations.tolist() still contains NumPy arrays!")
            print(f"DEBUG: Type of simulations_list: {type(simulations_list)}")

        if not isinstance(simulations_list, list):
            raise ValueError(f"Expected a list but got {type(simulations_list)}")
        # Ensure it is a list before passing to convert_prices_to_time_format
        if not isinstance(simulations, np.ndarray):
            raise ValueError(f"Unexpected data type for simulations: {type(simulations)}")

        predictions = convert_prices_to_time_format(
            simulations.tolist(), start_time, time_increment
        )
        
        for entry in predictions[:5]:  # Print a few entries
            print(f"DEBUG: Sample prediction entry: {entry}")
        print(f"DEBUG: Predictions seem to be correctly formatted: {type(predictions)}, Sample: {predictions[:2]}")
        return predictions
    except Exception as e:
        print(f"Error in generate_simulations: {e}")
        fallback_prediction = [[(start_time + i * time_increment, current_price) 
                              for i in range(time_length // time_increment + 1)]
                             for _ in range(num_simulations)]
        return fallback_prediction
