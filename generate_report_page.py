import base64
import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Read image as base64
with open('sbi_stock_prediction_graph.png', 'rb') as img_f:
    img_b64 = base64.b64encode(img_f.read()).decode('utf-8')

df = pd.read_csv('sbin_historical_2025_2026.csv')
df['Date'] = pd.to_datetime(df['Date']).dt.date
df = df.sort_values('Date').drop_duplicates(subset=['Date']).dropna(subset=['Close'])
df = df[df['Date'] <= datetime.date(2026, 3, 31)].reset_index(drop=True)
df['Day_Number'] = np.arange(1, len(df) + 1)

X = df[['Day_Number']]
y = df['Close']
model = LinearRegression().fit(X, y)
m = model.coef_[0]
c = model.intercept_
y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

NSE_HOLIDAYS_2026 = {
    datetime.date(2026, 4, 3), datetime.date(2026, 4, 14), datetime.date(2026, 5, 1),
    datetime.date(2026, 5, 28), datetime.date(2026, 6, 26), datetime.date(2026, 8, 28),
    datetime.date(2026, 10, 2), datetime.date(2026, 10, 20), datetime.date(2026, 11, 10),
    datetime.date(2026, 11, 24), datetime.date(2026, 12, 25)
}

def get_future_dates(start, n):
    res = []
    curr = start
    while len(res) < n:
        if curr.weekday() < 5 and curr not in NSE_HOLIDAYS_2026:
            res.append(curr)
        curr += datetime.timedelta(days=1)
    return res

future_days = 30
last_day = df['Day_Number'].iloc[-1]
dates = get_future_dates(datetime.date(2026, 4, 1), future_days)
day_nums = np.arange(last_day + 1, last_day + 1 + future_days)
preds = model.predict(pd.DataFrame({'Day_Number': day_nums}))

# Build clean HTML practical sheet
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SBI Stock Price Prediction - Machine Learning Practical</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    line-height: 1.6;
    color: #24292e;
    max-width: 920px;
    margin: 30px auto;
    padding: 0 20px;
    background-color: #f4f6f8;
  }}
  .practical-card {{
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 8px;
    padding: 36px 44px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
  }}
  h1 {{
    font-size: 25px;
    color: #0b4f8a;
    border-bottom: 2px solid #0b4f8a;
    padding-bottom: 10px;
    margin-top: 0;
  }}
  .subtitle {{
    font-size: 14px;
    color: #57606a;
    margin-bottom: 24px;
  }}
  h2 {{
    font-size: 18px;
    color: #0b4f8a;
    border-bottom: 1px solid #eaecef;
    padding-bottom: 6px;
    margin-top: 26px;
    margin-bottom: 12px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 14px;
  }}
  th, td {{
    border: 1px solid #d0d7de;
    padding: 8px 12px;
    text-align: left;
  }}
  th {{
    background-color: #f6f8fa;
    font-weight: 600;
  }}
  tr:nth-child(even) {{
    background-color: #fcfcfc;
  }}
  .equation-box {{
    background: #f1f8ff;
    border-left: 4px solid #0969da;
    padding: 14px 18px;
    font-size: 16px;
    margin: 14px 0;
    font-family: 'Consolas', 'Courier New', monospace;
  }}
  .metric-badge {{
    display: inline-block;
    background: #e8f3ff;
    color: #0550ae;
    border: 1px solid #b6d3f5;
    padding: 8px 14px;
    border-radius: 6px;
    margin-right: 10px;
    font-weight: bold;
    font-size: 14px;
  }}
  .graph-container {{
    text-align: center;
    margin: 22px 0;
  }}
  .graph-container img {{
    max-width: 100%;
    height: auto;
    border-radius: 6px;
    border: 1px solid #d0d7de;
  }}
  .disclaimer {{
    background-color: #fff8c5;
    border: 1px solid #d4a72c;
    border-radius: 6px;
    padding: 14px 18px;
    margin-top: 30px;
    font-size: 14px;
    color: #5c4300;
  }}
</style>
</head>
<body>
<div class="practical-card">
  <h1>SBI Stock Price Prediction Using Linear Regression</h1>
  <div class="subtitle">Machine Learning Practical &bull; Simple Single-Page Lab Record</div>

  <h2>1. Dataset Information</h2>
  <ul>
    <li><strong>Stock Symbol:</strong> SBIN (State Bank of India)</li>
    <li><strong>Exchange:</strong> National Stock Exchange of India (NSE India)</li>
    <li><strong>Training Period:</strong> 1 April 2025 to 31 March 2026</li>
    <li><strong>Total Valid Trading Days:</strong> {len(df)} sessions</li>
    <li><strong>Source:</strong> Actual daily historical records from NSE</li>
  </ul>

  <h2>2. First 5 Records of Historical Data</h2>
  <table>
    <thead>
      <tr><th>Date</th><th>Open (₹)</th><th>High (₹)</th><th>Low (₹)</th><th>Close (₹)</th><th>Volume</th><th>Day_Number</th></tr>
    </thead>
    <tbody>
"""

for idx, r in df.head(5).iterrows():
    html_content += f"""      <tr>
        <td>{r['Date']}</td>
        <td>{r['Open']:.2f}</td>
        <td>{r['High']:.2f}</td>
        <td>{r['Low']:.2f}</td>
        <td><strong>{r['Close']:.2f}</strong></td>
        <td>{int(r['Volume']):,}</td>
        <td>{r['Day_Number']}</td>
      </tr>\n"""

html_content += f"""    </tbody>
  </table>

  <h2>3 & 4. Independent (X) and Dependent (Y) Variables</h2>
  <p><strong>Independent Variable (X):</strong> <code>Day_Number</code> (Sequential integer index of trading days: 1, 2, 3, ... {len(df)})</p>
  <p><strong>Dependent Variable (Y):</strong> <code>Close Price</code> (Daily closing price of SBIN on NSE in ₹)</p>

  <h2>5 & 6. Linear Regression Equation, Slope and Intercept</h2>
  <div class="equation-box">
    Standard Form: y = mx + c<br>
    <strong>Fitted Equation: Predicted Close Price = {m:.4f} &times; Day_Number + {c:.4f}</strong>
  </div>
  <ul>
    <li><strong>Slope (m):</strong> <code>{m:.6f}</code> (Average daily closing price growth of ₹{m:.2f} per trading session)</li>
    <li><strong>Intercept (c):</strong> <code>{c:.6f}</code> (Base price constant in ₹)</li>
  </ul>

  <h2>7. Model Performance</h2>
  <div>
    <span class="metric-badge">R² Score: {r2:.4f} ({r2*100:.2f}%)</span>
    <span class="metric-badge">MAE: ₹{mae:.2f}</span>
    <span class="metric-badge">RMSE: ₹{rmse:.2f}</span>
  </div>

  <h2>8. Historical Data + Regression Graph</h2>
  <div class="graph-container">
    <img src="data:image/png;base64,{img_b64}" alt="Historical Data and Linear Regression Line" />
  </div>

  <h2>9. User-Selected Prediction Duration</h2>
  <p>Selected duration: <strong>{future_days} future trading days</strong> starting immediately from 1 April 2026.</p>

  <h2>10. Future Predicted SBI Prices ({future_days} Future Trading Days)</h2>
  <table>
    <thead>
      <tr><th>Future Day</th><th>Trading Date</th><th>Day Number</th><th>Predicted SBI Close</th></tr>
    </thead>
    <tbody>
"""

for i in range(future_days):
    html_content += f"""      <tr>
        <td>{i+1}</td>
        <td>{dates[i].strftime('%Y-%m-%d (%A)')}</td>
        <td>{day_nums[i]}</td>
        <td><strong>₹{preds[i]:.2f}</strong></td>
      </tr>\n"""

ex_day = day_nums[0]
ex_val = m * ex_day + c

html_content += f"""    </tbody>
  </table>

  <h2>11. Example Calculation (Manual Step-by-Step Verification)</h2>
  <div class="equation-box">
    For Future Day 1 ({dates[0].strftime('%Y-%m-%d')}) with Day_Number (x) = {ex_day}:<br><br>
    Predicted Price = m &times; Day_Number + c<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= ({m:.6f} &times; {ex_day}) + {c:.6f}<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {m*ex_day:.6f} + {c:.6f}<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <strong>₹{ex_val:.2f}</strong>
  </div>

  <h2>12. Conclusion & Explanation</h2>
  <p><strong>Linear Regression predicts a numerical value</strong>, making it suited to model continuous price trends over time.</p>
  <p>The model mapped: <strong>Trading Day Number &rarr; SBI Closing Price</strong> and fitted <code>y = {m:.4f}x + {c:.4f}</code>, capturing <strong>{r2*100:.2f}%</strong> of historical variance during the 2025–26 financial year.</p>

  <div class="disclaimer">
    <strong>Warning:</strong><br>
    "These are predictions generated by a Linear Regression model based on historical SBI prices. They are estimates only and are not guaranteed future market prices."
  </div>
</div>
</body>
</html>
"""

with open('sbi_practical_result.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Generated sbi_practical_result.html successfully.")
