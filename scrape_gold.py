# scrape_gold.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path

def check_if_update_needed(filename='gold_prices.csv'):
    """Check if we need to fetch new data by comparing last date with current date"""
    if not Path(filename).exists():
        return True  # File doesn't exist, need to fetch data
    
    try:
        # Read just the last line to get the most recent date
        with open(filename, 'r') as f:
            last_line = f.readlines()[-1].strip()
        
        # Extract the date from the CSV (assuming it's the first column)
        last_date_str = last_line.split(',')[0]
        last_date = datetime.strptime(last_date_str, '%Y-%m-%d').date()
        
        # Get current date (we'll compare dates only, ignoring time)
        current_date = datetime.now().date()
        
        # If last date is today or yesterday (market closed), no need to update
        # Also account for weekends when markets are closed
        if last_date == current_date:
            print("Data is already up to date (latest data is from today)")
            return False
        elif last_date == current_date - timedelta(days=1):
            # Markets might not have closed yet, but we already have yesterday's data
            print("Data is recent (latest data is from yesterday)")
            return False
        elif last_date.weekday() >= 5 and (current_date - last_date).days <= 2:
            # It's weekend and we have Friday's data
            print("Data is recent (markets closed on weekend)")
            return False
        
        print(f"Data needs update (last date: {last_date}, current date: {current_date})")
        return True
        
    except Exception as e:
        print(f"Error checking data freshness: {e}")
        return True  # If there's any error, proceed with update

def get_gold_prices(start_date="2001-01-01"):
    ticker = "GC=F"
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Fetching gold prices from {start_date} to {end_date}...")
    
    try:
        data = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False
        )
        
        if data.empty:
            print("Warning: No data from yfinance")
            return pd.DataFrame()
            
        data = data.reset_index()
        data['price_gr'] = data['Close'] / 31.1035  # Convert oz to grams
        result = data.rename(columns={'Date': 'datetime'})
        
        result.columns = [col[0] if isinstance(col, tuple) else str(col) for col in result.columns]
        result = result.rename(columns={
            'Close': 'Close',
            'High': 'High', 
            'Low': 'Low',
            'Open': 'Open',
            'Volume': 'Volume'
        })
        
        return result
        
    except Exception as e:
        print(f"Error fetching gold prices: {e}")
        return pd.DataFrame()

def save_to_csv(df, filename='gold_prices.csv'):
    """Save DataFrame to CSV file with selected columns"""
    columns_to_save = ['datetime', 'Close', 'High', 'Low', 'Open', 'Volume', 'price_gr']
    df[columns_to_save].to_csv(filename, index=False)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    # First check if we need to update
    if not check_if_update_needed():
        exit()
    
    # Hardcoded start date when run directly
    df = get_gold_prices("2001-01-01")
    if not df.empty:
        save_to_csv(df)
        print("\n=== Latest 5 Gold Prices ===")
        print(df.tail().to_string(index=False, float_format='%.5f'))
    else:
        print("No data was fetched")