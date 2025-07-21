import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import warnings
import subprocess
from pathlib import Path

warnings.filterwarnings('ignore')
START_DATE = "2020-01-01"

def update_model(model, new_data, scaler, look_back=60, epochs=5, batch_size=32):
    """
    Update existing model with new data incrementally.
    """
    # Preprocess new data
    new_data = validate_data(new_data)
    scaled_new_data = scaler.transform(new_data[['Close', 'High', 'Low', 'Open', 'Volume']].values)
    
    # Create sequences from new data only
    X_new, y_new = [], []
    for i in range(look_back, len(scaled_new_data)):
        X_new.append(scaled_new_data[i-look_back:i, :])
        y_new.append(scaled_new_data[i, 0])
    
    if len(X_new) == 0:
        print("Not enough new data to update.")
        return model
    
    X_new = np.array(X_new)
    y_new = np.array(y_new)
    
    # Train with a smaller learning rate for fine-tuning
    model.optimizer.lr = 0.0001  # Reduce learning rate
    model.fit(X_new, y_new, epochs=epochs, batch_size=batch_size, verbose=1)
    
    return model

def should_retrain(dataset_path='gold_prices.csv', model_path='gold_lstm_model.h5'):
    if not Path(model_path).exists():
        return True

    dataset_mtime = Path(dataset_path).stat().st_mtime
    model_mtime = Path(model_path).stat().st_mtime

    return dataset_mtime > model_mtime


def save_model(model, path='gold_lstm_model.h5'):
    model.save(path)
    print(f"Model saved to {path}")

def validate_data(df):
    required_cols = ['datetime', 'Close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Data must contain columns: {required_cols}")
    if len(df) < 60:
        raise ValueError("Insufficient data points (minimum 60 required)")
    return df.dropna(subset=['Close'])


def prepare_data(df, look_back=60):
    df = validate_data(df)

    features = ['Close', 'High', 'Low', 'Open', 'Volume']
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features].values)

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, :])
        y.append(scaled_data[i, 0])

    return np.array(X), np.array(y), scaler


def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(input_shape[1], input_shape[2])),
        Dropout(0.3),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae', 'mape'])
    return model


def train_model(X_train, y_train, epochs=50, batch_size=32):
    model = build_lstm_model(X_train.shape)
    print("\n=== Model Summary ===")
    model.summary()

    print("\n=== Training Metrics ===")
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )

    print("\n=== Final Training Results ===")
    print(f"Final Loss: {history.history['loss'][-1]:.6f}")
    print(f"Best Loss: {min(history.history['loss']):.6f}")
    print(f"Training stopped at epoch: {len(history.history['loss'])}")

    return model, history


def plot_training_results(model, history, X, y, scaler):
    plt.style.use('seaborn')
    plt.figure(figsize=(18, 12))
    
    # Plot 1: Training and validation loss
    plt.subplot(2, 2, 1)
    if 'loss' in history.history:
        plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linestyle='--')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Training metrics
    plt.subplot(2, 2, 2)
    metrics = ['loss', 'mae', 'mape']
    colors = ['blue', 'green', 'red']
    for i, metric in enumerate(metrics):
        if metric in history.history:
            plt.plot(history.history[metric], label=f'Train {metric.upper()}', color=colors[i], linewidth=2)
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val LOSS', color='orange', linestyle='--', linewidth=2)
    plt.title("Training Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Predictions vs actuals
    plt.subplot(2, 2, 3)
    predictions = model.predict(X, verbose=0)
    dummy = np.zeros((len(predictions), 5))
    dummy[:, 0] = predictions.flatten()
    predicted_prices = scaler.inverse_transform(dummy)[:, 0]

    dummy_true = np.zeros((len(y), 5))
    dummy_true[:, 0] = y
    true_prices = scaler.inverse_transform(dummy_true)[:, 0]

    plt.plot(true_prices, label='Actual Prices', color='blue')
    plt.plot(predicted_prices, label='Predicted Prices', color='orange')
    plt.title("Predicted vs Actual Prices")
    plt.xlabel("Time Step")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("training_results.png", dpi=300, bbox_inches='tight')
    plt.show()


def predict_future_prices(model, df, scaler, days_ahead, look_back=60):
    features = ['Close', 'High', 'Low', 'Open', 'Volume']
    last_sequence = scaler.transform(df[features].values[-look_back:])
    predictions = []

    for _ in range(days_ahead):
        x_input = last_sequence.reshape(1, look_back, len(features))
        pred = model.predict(x_input, verbose=0)
        predictions.append(pred[0, 0])

        new_row = np.copy(last_sequence[-1])
        new_row[0] = pred[0, 0]
        last_sequence = np.append(last_sequence[1:], [new_row], axis=0)

    dummy = np.zeros((len(predictions), len(features)))
    dummy[:, 0] = predictions
    predictions = scaler.inverse_transform(dummy)[:, 0]

    last_date = df['datetime'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]

    return pd.DataFrame({
        'date': future_dates,
        'predicted_price': predictions.flatten()
    })


def plot_results(df, predictions):
    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(df['datetime'], df['Close'],
             label='Historical Prices', color='royalblue', linewidth=2)
    ax1.plot(predictions['date'], predictions['predicted_price'],
             label='Forecast', color='forestgreen',
             marker='o', linestyle='--', linewidth=2)

    rolling_std = df['Close'].rolling(30).std().iloc[-1]
    ax1.fill_between(predictions['date'],
                     predictions['predicted_price'] - rolling_std,
                     predictions['predicted_price'] + rolling_std,
                     color='forestgreen', alpha=0.2, label='Confidence Band')

    ax1.set_title('Gold Price Prediction with Confidence Intervals', fontsize=16, pad=20)
    ax1.set_ylabel('Price (USD/oz)', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()

    ax2.plot(df['datetime'], df['SMA_50'], label='50-Day SMA', color='darkorange')
    ax2.plot(df['datetime'], df['SMA_200'], label='200-Day SMA', color='purple')
    ax2.set_title('Technical Indicators', fontsize=14, pad=15)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Moving Averages', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return plt

def main():
    try:
        # Run the scraper first with our hardcoded start date
        print(f"Checking for new data since {START_DATE}...")
        scrape_cmd = f"python scrape_gold.py"
        subprocess.run(scrape_cmd, shell=True, check=True)

        # Load data
        df = pd.read_csv('gold_prices.csv', skiprows=1, header=None,
                         names=['datetime', 'Close', 'High', 'Low', 'Open', 'Volume', 'price_gr'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Filter data to only include dates after our start date
        df = df[df['datetime'] >= pd.to_datetime(START_DATE)]
        
        if should_retrain() or not Path('gold_lstm_model.h5').exists():
            # Check if we have an existing model
            if Path('gold_lstm_model.h5').exists():
                # Prepare all data to get the scaler
                X, y, scaler = prepare_data(df)
                
                # Find the last date in the training data
                last_trained_date = df['datetime'].max()
                new_data = df[df['datetime'] > last_trained_date]
                
                if len(new_data) > 0:
                    print(f"Found {len(new_data)} new data points. Updating model incrementally...")
                    model = load_model('gold_lstm_model.h5')
                    model = update_model(model, new_data, scaler)
                    save_model(model)
                else:
                    print("No new data found. Using existing model.")
                    model = load_model('gold_lstm_model.h5')
            else:
                print("Training from scratch...")
                X, y, scaler = prepare_data(df)
                model, history = train_model(X, y, 100)
                save_model(model)
                
                print("\n=== Model Evaluation ===")
                metrics = model.evaluate(X, y, verbose=0)
                metric_names = model.metrics_names

                for name, value in zip(metric_names, metrics):
                    print(f"{name.upper():<10}: {value:.6f}")

                try:
                    mape_index = metric_names.index('mape')
                    mape = metrics[mape_index]
                    accuracy = 100 - mape
                    print(f"ACCURACY : {accuracy:.2f}% (derived from 100 - MAPE)")
                except ValueError:
                    print("MAPE not found in metrics. Cannot calculate accuracy.")

                plot_training_results(model, history, X, y, scaler)
        else:
            print("Loading existing model...")
            model = load_model('gold_lstm_model.h5')
            # We need to prepare data to get the scaler for predictions
            X, y, scaler = prepare_data(df)

        # Prediction menu and plotting (keep the existing code)
        timeframes = {1: 3, 2: 7, 3: 14, 4: 21, 5: 30}
        while True:
            print("\nSelect prediction timeframe:")
            print("1. 3 days")
            print("2. 1 week (7 days)")
            print("3. 2 weeks (14 days)")
            print("4. 3 weeks (21 days)")
            print("5. 1 month (30 days)")

            choice = input("Enter choice (1-5): ")
            try:
                days_ahead = timeframes.get(int(choice), 30)
                break
            except (ValueError, KeyError):
                print("Invalid input. Please enter a number between 1-5")

        predictions = predict_future_prices(model, df, scaler, days_ahead)
        predictions.to_csv('gold_lstm_predictions.csv', index=False)

        print(f"\n=== Gold Price Predictions ({days_ahead} days) ===")
        print(predictions.to_string(index=False))

        plot = plot_results(df, predictions)
        plot.savefig('gold_price_prediction.png', dpi=300, bbox_inches='tight')
        plot.show()

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()