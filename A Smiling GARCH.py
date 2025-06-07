import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta

class NGARCH:
    def __init__(self, returns):
        # Convert returns to numpy array and ensure it's 1D
        self.returns = np.asarray(returns).flatten()
        self.n = len(self.returns)
        # Calculate initial mean and variance
        self.initial_mean = float(np.mean(self.returns))
        self.initial_variance = float(np.var(self.returns, ddof=1))
        
    def log_likelihood(self, params):
        """Calculate the log-likelihood for NGARCH(1,1) model with mean return"""
        mu, omega, alpha, beta, theta = params
        sigma2 = np.zeros(self.n)
        sigma2[0] = self.initial_variance
        
        # Calculate standardized residuals
        epsilon = self.returns - mu
        
        for t in range(1, self.n):
            sigma2[t] = omega + alpha * (epsilon[t-1] - theta * np.sqrt(sigma2[t-1]))**2 + beta * sigma2[t-1]
            
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + epsilon**2 / sigma2)
        return -log_likelihood  # Negative for minimization
    
    def estimate(self):
        """Estimate NGARCH parameters using maximum likelihood"""
        # Initial parameter guesses [mu, omega, alpha, beta, theta]
        initial_params = [self.initial_mean, 0.01, 0.1, 0.8, 0.5]
        
        # Parameter bounds
        bounds = [(None, None), (1e-6, None), (1e-6, 1), (1e-6, 1), (None, None)]
        
        # Optimize
        result = minimize(self.log_likelihood, initial_params, bounds=bounds, method='L-BFGS-B')
        
        if not result.success:
            print("Warning: Optimization did not converge successfully")
            print(f"Message: {result.message}")
        
        return result.x
    
    def get_volatility_forecast(self, params, horizon=1):
        """Get volatility forecast for given horizon"""
        mu, omega, alpha, beta, theta = params
        sigma2 = np.zeros(self.n)
        sigma2[0] = self.initial_variance
        
        # Calculate standardized residuals
        epsilon = self.returns - mu
        
        for t in range(1, self.n):
            sigma2[t] = omega + alpha * (epsilon[t-1] - theta * np.sqrt(sigma2[t-1]))**2 + beta * sigma2[t-1]
            
        # Forecast future volatility
        forecast = np.zeros(horizon)
        forecast[0] = sigma2[-1]
        
        for t in range(1, horizon):
            forecast[t] = omega + (alpha * (1 + theta**2) + beta) * forecast[t-1]
            
        return np.sqrt(forecast)

def get_option_data(symbol):
    """Get option data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        
        # Get available expiration dates
        expirations = stock.options
        if not expirations:
            print(f"No option expiration dates available for {symbol}")
            return None, None
        
        # Find the nearest expiration date that's at least 7 days away
        today = datetime.now().date()
        valid_expirations = []
        for exp in expirations:
            exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
            days_to_exp = (exp_date - today).days
            if 7 <= days_to_exp <= 30:  # Look for options expiring between 1 week and 1 month
                valid_expirations.append(exp)
        
        if not valid_expirations:
            print("No suitable expiration dates found (looking for options expiring between 1 week and 1 month)")
            return None, None
            
        # Use the nearest valid expiration date
        expiration_date = valid_expirations[0]
        print(f"Using expiration date: {expiration_date} ({(datetime.strptime(expiration_date, '%Y-%m-%d').date() - today).days} days to expiration)")
        
        options = stock.option_chain(expiration_date)
        
        # Create new DataFrames instead of modifying slices
        calls = options.calls.copy()
        puts = options.puts.copy()
        
        if calls.empty or puts.empty:
            print(f"No option data available for expiration {expiration_date}")
            return None, None
            
        # Filter out options with no volume or open interest
        calls = calls[(calls['volume'] > 0) | (calls['openInterest'] > 0)].copy()
        puts = puts[(puts['volume'] > 0) | (puts['openInterest'] > 0)].copy()
        
        if calls.empty or puts.empty:
            print("No active options found after filtering")
            return None, None
            
        # Additional filtering for data quality
        calls = calls[calls['lastPrice'] > 0.01]  # Filter out very low-priced options
        puts = puts[puts['lastPrice'] > 0.01]
        
        if calls.empty or puts.empty:
            print("No valid options found after price filtering")
            return None, None
            
        # Calculate implied volatility
        calls.loc[:, 'implied_volatility'] = calls['impliedVolatility']
        puts.loc[:, 'implied_volatility'] = puts['impliedVolatility']
        
        # Filter out any NaN or infinite implied volatilities
        calls = calls[np.isfinite(calls['implied_volatility'])]
        puts = puts[np.isfinite(puts['implied_volatility'])]
        
        # Filter out unreasonable implied volatilities
        calls = calls[(calls['implied_volatility'] > 0) & (calls['implied_volatility'] < 5)]
        puts = puts[(puts['implied_volatility'] > 0) & (puts['implied_volatility'] < 5)]
        
        if calls.empty or puts.empty:
            print("No options with valid implied volatilities found")
            return None, None
            
        print(f"Found {len(calls)} valid call options and {len(puts)} valid put options")
        return calls, puts
    except Exception as e:
        print(f"Error fetching option data: {e}")
        return None, None

def black_scholes_iv(S, K, T, r, option_price, option_type='call'):
    """Calculate implied volatility using Black-Scholes model"""
    def black_scholes_price(sigma):
        # Add numerical stability checks
        if sigma <= 0 or T <= 0:
            return float('inf')
            
        try:
            d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                
            return price - option_price
        except:
            return float('inf')
    
    # Use Newton-Raphson method to find implied volatility
    sigma = 0.2  # Initial guess
    max_iter = 100
    tolerance = 1e-6
    
    for i in range(max_iter):
        f = black_scholes_price(sigma)
        if abs(f) < tolerance:
            break
            
        f_prime = (black_scholes_price(sigma + 0.0001) - f) / 0.0001
        if f_prime == 0:
            break
            
        sigma = sigma - f / f_prime
        if sigma <= 0:
            sigma = 0.0001  # Ensure positive volatility
            
        # Check for convergence
        if i == max_iter - 1:
            print(f"Warning: Newton-Raphson did not converge for K={K}, T={T}, price={option_price}")
            return None
            
    return sigma

def extract_expiration_date(contract_symbol):
    """Extract expiration date from contract symbol (e.g., NVDA240315C00100000)"""
    # The date is in the format YYMMDD starting after the ticker symbol
    date_str = contract_symbol[4:10]  # Get YYMMDD part
    year = 2000 + int(date_str[:2])   # Convert YY to YYYY
    month = int(date_str[2:4])
    day = int(date_str[4:6])
    return datetime(year, month, day)

def main():
    try:
        # Get NVDA data
        print("Downloading NVDA data...")
        nvda = yf.download('NVDA', start='2020-01-01', end=datetime.now())
        if nvda.empty:
            print("Yahoo Finance download failed or returned empty. Trying to load from local CSV 'nvda_data.csv'...")
            try:
                nvda = pd.read_csv('nvda_data.csv', index_col=0, parse_dates=True)
                print("Loaded NVDA data from 'nvda_data.csv'.")
            except Exception as e:
                print("Failed to load NVDA data from local CSV. Exiting...")
                print(f"Error: {e}")
                return
        returns = np.log(nvda['Close'] / nvda['Close'].shift(1)).dropna()
        if returns.empty:
            print("No returns data available. Exiting...")
            return
        
        # Estimate NGARCH model
        print("Estimating NGARCH parameters...")
        ngarch = NGARCH(returns)
        params = ngarch.estimate()
        
        print("\nNGARCH Parameters:")
        print(f"Mean Return (μ): {params[0]:.6f}")
        print(f"Omega (ω): {params[1]:.6f}")
        print(f"Alpha (α): {params[2]:.6f}")
        print(f"Beta (β): {params[3]:.6f}")
        print(f"Theta (θ): {params[4]:.6f}")
        
        # Get option data
        print("\nFetching option data...")
        calls, puts = get_option_data('NVDA')
        
        if calls is None or puts is None:
            print("Failed to fetch option data. Exiting...")
            return
            
        # Print DataFrame structure for debugging
        print("\nCalls DataFrame columns:", calls.columns.tolist())
        print("First row of calls DataFrame:")
        print(calls.iloc[0])
        
        # Calculate NGARCH implied volatilities
        S = float(nvda['Close'].iloc[-1])  # Ensure S is a float for formatting
        r = 0.05  # Risk-free rate
        
        # Get the expiration date from the contract symbol
        try:
            expiration_date = extract_expiration_date(calls.iloc[0]['contractSymbol'])
            print(f"Extracted expiration date from contract symbol: {expiration_date.strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"Error extracting expiration date: {e}")
            print("Contract symbol:", calls.iloc[0]['contractSymbol'])
            return
                
        current_date = datetime.now()
        T = (expiration_date - current_date).total_seconds() / (365.25 * 24 * 3600)  # Convert to years
        
        if T <= 0:
            print(f"Error: Invalid time to expiration: {T} years")
            print(f"Expiration date: {expiration_date}")
            print(f"Current date: {current_date}")
            return
            
        print(f"Current stock price: ${S:.2f}")
        print(f"Time to expiration: {T:.4f} years")
        print(f"Expiration date: {expiration_date.strftime('%Y-%m-%d')}")
        
        print("Calculating implied volatilities...")
        # Create a new column for NGARCH IV
        calls = calls.copy()
        
        # Calculate implied volatilities with error handling
        valid_ivs = []
        for idx, row in calls.iterrows():
            try:
                iv = black_scholes_iv(S, row['strike'], T, r, row['lastPrice'], 'call')
                if iv is not None and 0 < iv < 5:  # Add reasonable bounds for IV
                    valid_ivs.append((idx, iv))
            except Exception as e:
                print(f"Error calculating IV for strike {row['strike']}: {e}")
                continue
        
        if not valid_ivs:
            print("No valid implied volatilities calculated. Exiting...")
            return
            
        # Update the DataFrame with valid implied volatilities
        for idx, iv in valid_ivs:
            calls.loc[idx, 'ngarch_iv'] = iv
        
        # Filter out any NaN or infinite values
        calls = calls[np.isfinite(calls['ngarch_iv'])]
        
        if calls.empty:
            print("No valid implied volatilities calculated. Exiting...")
            return
        
        # Plot results
        print("Generating plot...")
        plt.figure(figsize=(12, 6))
        plt.scatter(calls['strike'], calls['implied_volatility'], 
                    label='Market IV', alpha=0.5)
        plt.scatter(calls['strike'], calls['ngarch_iv'], 
                    label='NGARCH IV', alpha=0.5)
        plt.xlabel('Strike Price')
        plt.ylabel('Implied Volatility')
        plt.title('NVDA Implied Volatility Smile: Market vs NGARCH')
        plt.legend()
        plt.grid(True)
        plt.savefig('ngarch_iv_comparison.png')
        plt.show()
        plt.close()
        
        # Save results to CSV
        print("Saving results to CSV...")
        results = pd.DataFrame({
            'strike': calls['strike'],
            'market_iv': calls['implied_volatility'],
            'ngarch_iv': calls['ngarch_iv']
        })
        results.to_csv('ngarch_iv_results.csv', index=False)
        
        print("\nAnalysis complete! Results saved to 'ngarch_iv_results.csv' and 'ngarch_iv_comparison.png'")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 