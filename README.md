# Gold Price Prediction Project

This project provides tools to scrape gold price data and predict future gold prices using LSTM neural networks.

## Features

* **Data Scraping**: Automatically fetches gold price data from Yahoo Finance
* **Data Validation**: Checks if data needs to be updated
* **Machine Learning**: Uses LSTM model for price prediction
* **Visualization**: Generates plots of predictions and technical indicators
* **Incremental Learning**: Can update model with new data without retraining from scratch

## Requirements

* Python 3.8+
* Required packages (see requirements.txt):
  

```
  requests==2.31.0
  beautifulsoup4==4.12.2
  pandas==2.0.3
  scikit-learn==1.3.0
  numpy==1.24.3
  tensorflow==2.12.0
  matplotlib==3.7.2
  keras==2.12.0
  yfinance
  ```

## Installation

1. Clone this repository:
   

```bash
   git clone https://github.com/AvuxDemons/GoldPricePredict.git
   cd GoldPricePredict
   ```

2. Install dependencies:
   

```bash
   pip install -r requirements.txt
   ```

## Usage

1. First run the scraper to get latest gold prices:
   

```bash
   python scrape_gold.py
   ```

2. Then run the predictor:
   

```bash
   python predictor.py
   ```

3. Follow the on-screen prompts to select prediction timeframe (3 days to 1 month)

## File Structure

* `scrape_gold.py`: Script to fetch and update gold price data
* `predictor.py`: Main prediction script with LSTM model
* `gold_prices.csv`: Historical gold price data
* `gold_lstm_model.h5`: Saved LSTM model (created after first run)
* `gold_lstm_predictions.csv`: Latest predictions (created after running predictor)
* `training_results.png`: Training metrics visualization
* `gold_price_prediction.png`: Prediction visualization

## How It Works

1. **Data Collection**:
   - Uses yfinance to fetch gold price data (GC=F ticker)
   - Converts prices from oz to grams
   - Saves data to CSV with OHLCV format

2. **Model Training**:
   - Uses 3-layer LSTM architecture with dropout
   - Normalizes data using MinMaxScaler
   - Implements early stopping to prevent overfitting

3. **Prediction**:
   - Predicts prices for selected timeframe (3-30 days)
   - Shows confidence intervals based on historical volatility
   - Includes technical indicators (50-day and 200-day SMA)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
