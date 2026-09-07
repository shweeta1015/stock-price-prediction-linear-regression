"""
Machine Learning Practical: SBI Stock Price Prediction Using Linear Regression
Target: Predict future SBI (SBIN) share closing price using Linear Regression.
Market: NSE India (Ticker: SBIN.NS)
Dataset: 1 April 2025 to 31 March 2026
"""

import sys
import os
import datetime

# Ensure utf-8 encoding on Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf

# NSE India trading holidays in 2026
NSE_HOLIDAYS_2026 = {
    datetime.date(2026, 4, 3),   # Good Friday
    datetime.date(2026, 4, 14),  # Dr. Baba Saheb Ambedkar Jayanti
    datetime.date(2026, 5, 1),   # Maharashtra Day
    datetime.date(2026, 5, 28),  # Bakri Id / Eid-ul-Adha
    datetime.date(2026, 6, 26),  # Muharram
    datetime.date(2026, 8, 28),  # Milad-un-Nabi
    datetime.date(2026, 10, 2),  # Mahatma Gandhi Jayanti
    datetime.date(2026, 10, 20), # Dussehra
    datetime.date(2026, 11, 10), # Diwali Balipratipada
    datetime.date(2026, 11, 24), # Guru Nanak Jayanti
    datetime.date(2026, 12, 25), # Christmas
}

def fetch_and_clean_data(csv_file="sbin_historical_2025_2026.csv"):
    """
    Fetch real historical SBI stock data from NSE for FY 2025-26.
    Cache locally to avoid repeated network requests.
    """
    if os.path.exists(csv_file):
        print(f"Loading cached real historical data from {csv_file}...")
        df = pd.read_csv(csv_file)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
    else:
        print("Fetching real historical SBI (SBIN) data from NSE (1 April 2025 - 31 March 2026)...")
        raw = yf.download('SBIN.NS', start='2025-04-01', end='2026-04-01')
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.to_csv(csv_file, index=False)
        print(f"Saved real historical data to {csv_file}")

    # Data Cleaning
    # 1. Sort chronologically
    df = df.sort_values('Date').reset_index(drop=True)
    # 2. Remove duplicate records
    df = df.drop_duplicates(subset=['Date']).reset_index(drop=True)
    # 3. Handle missing values
    df = df.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume']).reset_index(drop=True)
    # 4. Filter strictly up to 31 March 2026
    df = df[df['Date'] <= datetime.date(2026, 3, 31)].reset_index(drop=True)
    # 5. Create Day_Number (1, 2, 3, ...)
    df['Day_Number'] = np.arange(1, len(df) + 1)

    return df

def generate_future_trading_dates(start_date, n_days):
    """
    Generate actual future NSE trading dates excluding weekends and NSE holidays.
    """
    trading_dates = []
    current = start_date
    while len(trading_dates) < n_days:
        if current.weekday() < 5 and current not in NSE_HOLIDAYS_2026:
            trading_dates.append(current)
        current += datetime.timedelta(days=1)
    return trading_dates

def run_practical(future_days=30):
    print("=" * 80)
    print("SBI Stock Price Prediction Using Linear Regression (Machine Learning Practical)")
    print("=" * 80)

    # 1. Dataset Information
    df = fetch_and_clean_data()
    print("\n--- 1. DATASET INFORMATION ---")
    print(f"Stock Symbol       : SBIN (State Bank of India)")
    print(f"Market             : NSE India")
    print(f"Training Period    : 1 April 2025 to 31 March 2026")
    print(f"Total Trading Days : {len(df)}")
    print(f"Features Available : {list(df.columns)}")

    # 2. First 5 Records
    print("\n--- 2. FIRST 5 RECORDS ---")
    print(df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Day_Number']].head().to_string(index=False))

    # 3 & 4. Independent (X) and Dependent (Y) Variables
    print("\n--- 3. INDEPENDENT VARIABLE (X) ---")
    print("X = Day_Number (Sequential trading day counter: 1, 2, 3, ... N)")

    print("\n--- 4. DEPENDENT VARIABLE (Y) ---")
    print("Y = Close Price (Official NSE closing price of SBIN in ₹)")

    # 5 & 6. Linear Regression Model Training
    X = df[['Day_Number']]
    y = df['Close']

    model = LinearRegression()
    model.fit(X, y)

    slope_m = model.coef_[0]
    intercept_c = model.intercept_
    y_pred_hist = model.predict(X)

    print("\n--- 5. REGRESSION EQUATION ---")
    print(f"Formula            : y = mx + c")
    print(f"Fitted Equation    : y = {slope_m:.4f}x + {intercept_c:.4f}")

    print("\n--- 6. SLOPE AND INTERCEPT ---")
    print(f"Slope (m)          : {slope_m:.6f}  (Average price change per trading day in ₹)")
    print(f"Intercept (c)      : {intercept_c:.6f}  (Theoretical base price at Day 0 in ₹)")

    # 7. Model Performance
    r2 = r2_score(y, y_pred_hist)
    mae = mean_absolute_error(y, y_pred_hist)
    rmse = np.sqrt(mean_squared_error(y, y_pred_hist))

    print("\n--- 7. MODEL PERFORMANCE ---")
    print(f"R² Score           : {r2:.4f} ({r2*100:.2f}% of price variance explained by trading day)")
    print(f"MAE (Mean Absolute Error) : ₹{mae:.2f}")
    print(f"RMSE (Root Mean Sq Error) : ₹{rmse:.2f}")

    # 8 & 9. Prediction for User-Selected Duration
    print(f"\n--- 9. USER-SELECTED PREDICTION DURATION ---")
    print(f"Selected Duration  : {future_days} future trading days")

    last_day_num = df['Day_Number'].iloc[-1]
    # First trading day after 31 March 2026 is 1 April 2026
    start_future_date = datetime.date(2026, 4, 1)
    future_dates = generate_future_trading_dates(start_future_date, future_days)

    future_day_numbers = np.arange(last_day_num + 1, last_day_num + 1 + future_days)
    future_X = pd.DataFrame({'Day_Number': future_day_numbers})
    future_preds = model.predict(future_X)

    future_table = pd.DataFrame({
        'Future Day': np.arange(1, future_days + 1),
        'Trading Date': [d.strftime('%Y-%m-%d') for d in future_dates],
        'Day Number': future_day_numbers,
        'Predicted SBI Close': [f"₹{p:.2f}" for p in future_preds]
    })

    print(f"\n--- 10. FUTURE PREDICTED SBI PRICES (NEXT {future_days} TRADING DAYS) ---")
    print(future_table.to_string(index=False))

    # 11. Example Calculation
    ex_day_idx = 0
    ex_future_day = 1
    ex_date = future_dates[0].strftime('%Y-%m-%d')
    ex_day_num = future_day_numbers[0]
    ex_pred_val = future_preds[0]

    print("\n--- 11. EXAMPLE CALCULATION ---")
    print(f"For Future Day {ex_future_day} ({ex_date}):")
    print(f"Day Number (x) = {ex_day_num}")
    print(f"Regression Equation : Predicted Price = m × Day_Number + c")
    print(f"Step-by-step substitution:")
    print(f"Predicted Price = ({slope_m:.6f} × {ex_day_num}) + {intercept_c:.6f}")
    print(f"                = {slope_m * ex_day_num:.6f} + {intercept_c:.6f}")
    print(f"                = ₹{ex_pred_val:.2f}")

    # 8. Historical Data + Regression Graph
    plt.figure(figsize=(12, 6))
    # Historical actual prices
    plt.scatter(df['Day_Number'], df['Close'], color='#1f77b4', s=16, alpha=0.7, label='Actual Historical SBI Close')
    # Fitted regression line on training data
    plt.plot(df['Day_Number'], y_pred_hist, color='#2ca02c', linewidth=2, label=f'Linear Regression Fit (y = {slope_m:.2f}x + {intercept_c:.2f})')
    # Future predictions line
    plt.plot(future_day_numbers, future_preds, color='#d62728', linestyle='--', linewidth=2.5, marker='o', markersize=3, label=f'Future Predictions ({future_days} Trading Days)')

    # Boundary separator
    plt.axvline(x=last_day_num, color='gray', linestyle=':', label='Training End (31 March 2026)')

    plt.title('SBI (SBIN) Stock Price Prediction Using Linear Regression', fontsize=14, pad=12, fontweight='bold')
    plt.xlabel('Trading Day Sequence (Day_Number)', fontsize=12)
    plt.ylabel('SBI Closing Price (₹)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()

    graph_filename = "sbi_stock_prediction_graph.png"
    plt.savefig(graph_filename, dpi=200)
    plt.close()
    print(f"\n--- 8. GRAPH SAVED ---")
    print(f"Graph successfully saved as '{graph_filename}'.")

    # 10. Important Explanation
    print("\n--- 10. IMPORTANT EXPLANATION ---")
    print("Linear Regression predicts a continuous numerical value, making it suitable for")
    print("modeling price trends over time. In this practical, the model learns the statistical")
    print("relationship between the trading day sequence (Day_Number) and the SBI closing price:")
    print("                         Trading Day Number → SBI Closing Price")
    print(f"It fits the line: y = {slope_m:.4f}x + {intercept_c:.4f}")
    print("This line is then extended forward into future trading days to estimate future prices.")

    # 12. Conclusion & Final Warning
    print("\n--- 12. CONCLUSION & FINAL WARNING ---")
    print("Conclusion:")
    print(f"- Over FY 2025-26, SBI demonstrated an upward linear trend with a slope of ₹{slope_m:.2f} per day.")
    print(f"- The model explains {r2*100:.2f}% of the variance in the historical closing prices.")
    print(f"- Linear extrapolation provides a baseline trend estimation for subsequent trading sessions.")
    print("\nFinal Warning:")
    print('"These are predictions generated by a Linear Regression model based on historical SBI prices.')
    print('They are estimates only and are not guaranteed future market prices."')
    print("=" * 80)

if __name__ == "__main__":
    # Allow command-line argument or interactive input
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except ValueError:
            days = 30
    else:
        try:
            val = input("Enter number of future trading days to predict (e.g., 10, 20, 30, 60, 90, 120) [Default: 30]: ").strip()
            days = int(val) if val else 30
        except (EOFError, ValueError):
            days = 30

    run_practical(future_days=days)
