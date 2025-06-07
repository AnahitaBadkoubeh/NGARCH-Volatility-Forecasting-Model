# NGARCH Model for Volatility Smile Analysis

This project implements a Non-linear GARCH (NGARCH) model to analyze and compare implied volatility smiles in options markets, specifically focusing on NVIDIA (NVDA) stock options.

## Overview

The NGARCH model is an extension of the traditional GARCH model that incorporates leverage effects and non-linear dynamics in volatility. This implementation:

- Estimates NGARCH parameters from historical stock returns
- Calculates implied volatilities using the Black-Scholes model
- Compares market-implied volatilities with NGARCH-predicted volatilities
- Visualizes the volatility smile for both market and model predictions

## Model Specification

The NGARCH(1,1) model is specified as:

σ²ₜ = ω + α(εₜ₋₁ - θ√σ²ₜ₋₁)² + βσ²ₜ₋₁

Where:
- σ²ₜ is the conditional variance at time t
- ω is the constant term
- α is the ARCH term coefficient
- β is the GARCH term coefficient
- θ is the leverage effect parameter
- εₜ₋₁ is the lagged return innovation

## Features

- **Data Handling**:
  - Downloads historical stock data from Yahoo Finance
  - Fallback to local CSV data if API rate limits are hit
  - Fetches current options data for volatility analysis

- **Parameter Estimation**:
  - Maximum likelihood estimation of NGARCH parameters
  - Robust error handling and parameter bounds
  - Convergence checks for optimization

- **Volatility Analysis**:
  - Black-Scholes implied volatility calculations
  - NGARCH volatility forecasts
  - Volatility smile visualization
  - Data quality filters for reliable results

## Requirements

```python
numpy
pandas
matplotlib
scipy
yfinance
```

## Usage

1. Run the main script:
```python
python A Smiling GARCH.py
```

2. The script will:
   - Download NVDA historical data
   - Estimate NGARCH parameters
   - Fetch current options data
   - Calculate and compare implied volatilities
   - Generate visualization plots
   - Save results to CSV

## Output

The script generates:
- `ngarch_iv_comparison.png`: Plot comparing market and NGARCH implied volatilities
- `ngarch_iv_results.csv`: Detailed results including strike prices and both market and model implied volatilities

## Model Parameters

The estimated parameters include:
- μ (Mean Return)
- ω (Constant Term)
- α (ARCH Term)
- β (GARCH Term)
- θ (Leverage Effect)

## Error Handling

The implementation includes robust error handling for:
- API rate limits
- Data availability
- Numerical stability
- Parameter estimation convergence
- Invalid implied volatility calculations

## Notes

- The model uses a 1-month lookback period for options data
- Implied volatilities are filtered for reasonable ranges (0 < IV < 5)
- The risk-free rate is set to 5% (adjustable in the code)

## Future Improvements

Potential enhancements:
- Add more sophisticated volatility surface modeling
- Implement additional GARCH variants
- Add parameter stability analysis
- Include more sophisticated options data filtering
- Add backtesting capabilities

## License

This project is open source and available under the MIT License.

## Author

Anahita Badkoubeh Hezaveh

## Acknowledgments

- Based on the NGARCH model by Engle and Ng (1993)
- Uses Yahoo Finance API for market data
- Implements Black-Scholes model for options pricing 
