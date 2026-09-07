import os
import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="NSE Stock Price Prediction - Linear Regression")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Company Registry
COMPANIES = {
    "SBIN": {"name": "State Bank of India", "symbol": "SBIN", "file": "sbin_historical_2025_2026.csv"},
    "RELIANCE": {"name": "Reliance Industries", "symbol": "RELIANCE", "file": "reliance_historical_2025_2026.csv"},
    "TCS": {"name": "Tata Consultancy Services", "symbol": "TCS", "file": "tcs_historical_2025_2026.csv"},
    "INFY": {"name": "Infosys", "symbol": "INFY", "file": "infy_historical_2025_2026.csv"},
    "HDFCBANK": {"name": "HDFC Bank", "symbol": "HDFCBANK", "file": "hdfcbank_historical_2025_2026.csv"},
    "ICICIBANK": {"name": "ICICI Bank", "symbol": "ICICIBANK", "file": "icicibank_historical_2025_2026.csv"},
    "ITC": {"name": "ITC", "symbol": "ITC", "file": "itc_historical_2025_2026.csv"},
    "LT": {"name": "Larsen & Toubro", "symbol": "LT", "file": "lt_historical_2025_2026.csv"},
    "BHARTIARTL": {"name": "Bharti Airtel", "symbol": "BHARTIARTL", "file": "bhartiartl_historical_2025_2026.csv"},
    "TATAMOTORS": {"name": "Tata Motors", "symbol": "TATAMOTORS", "file": "tatamotors_historical_2025_2026.csv"}
}

# Preload and train models for all 10 companies
MODELS = {}
COMPANY_DATA = {}

BASE_DIR = os.path.dirname(__file__)

for code, info in COMPANIES.items():
    fpath = os.path.join(BASE_DIR, info["file"])
    df = pd.read_csv(fpath)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).dropna(subset=["Close"])
    df = df[df["Date"] <= datetime.date(2026, 3, 31)].reset_index(drop=True)
    df["Day_Number"] = np.arange(1, len(df) + 1)

    X = df[["Day_Number"]]
    y = df["Close"]
    model = LinearRegression()
    model.fit(X, y)
    df["Trend"] = model.predict(X)

    MODELS[code] = {
        "model": model,
        "last_day_num": int(df["Day_Number"].max()),
        "last_date": df["Date"].max()
    }

    COMPANY_DATA[code] = {
        "id": code,
        "name": info["name"],
        "symbol": info["symbol"],
        "dates": [d.strftime("%d-%m-%Y") for d in df["Date"]],
        "raw_dates": [d.strftime("%Y-%m-%d") for d in df["Date"]],
        "actual_prices": [round(float(p), 2) for p in df["Close"]],
        "trend_prices": [round(float(p), 2) for p in df["Trend"]],
        "last_date": df["Date"].max().strftime("%Y-%m-%d"),
        "min_future_date": "2026-04-01"
    }

# Official NSE India market holidays in 2026
NSE_HOLIDAYS_2026 = {
    datetime.date(2026, 4, 3),   # Good Friday
    datetime.date(2026, 4, 14),  # Dr. Ambedkar Jayanti
    datetime.date(2026, 5, 1),   # Maharashtra Day
    datetime.date(2026, 5, 28),  # Bakri Id / Eid-ul-Adha
    datetime.date(2026, 6, 26),  # Muharram
    datetime.date(2026, 8, 15),  # Independence Day
    datetime.date(2026, 8, 28),  # Milad-un-Nabi
    datetime.date(2026, 10, 2),  # Mahatma Gandhi Jayanti
    datetime.date(2026, 10, 20), # Dussehra
    datetime.date(2026, 11, 10), # Diwali Balipratipada
    datetime.date(2026, 11, 24), # Guru Nanak Jayanti
    datetime.date(2026, 12, 25), # Christmas
}

def count_future_trading_days(target_date: datetime.date) -> int:
    curr = datetime.date(2026, 4, 1)
    trading_count = 0
    while curr <= target_date:
        if curr.weekday() < 5 and curr not in NSE_HOLIDAYS_2026:
            trading_count += 1
        curr += datetime.timedelta(days=1)
    return trading_count

@app.get("/api/companies")
def get_companies():
    return [
        {"id": code, "name": info["name"], "symbol": info["symbol"]}
        for code, info in COMPANIES.items()
    ]

@app.get("/api/company_data")
def get_company_data(company: str = "SBIN"):
    code = company.upper()
    if code not in COMPANY_DATA:
        return JSONResponse(status_code=404, content={"error": "Company not found"})
    return COMPANY_DATA[code]

@app.get("/api/predict")
def predict_stock_price(company: str, date: str):
    code = company.upper()
    if code not in MODELS:
        return JSONResponse(status_code=404, content={"success": False, "error": "Company not found."})

    try:
        selected_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Invalid date format. Please use YYYY-MM-DD."}
        )

    # 1. Date must be in future after 31 March 2026
    if selected_date <= datetime.date(2026, 3, 31):
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Please select a future date starting from 1 April 2026."}
        )

    # 2. Weekend check (Saturday / Sunday)
    if selected_date.weekday() in (5, 6):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Selected date is not an NSE trading day. Please select another date."
            }
        )

    # 3. NSE Holiday check
    if selected_date in NSE_HOLIDAYS_2026:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Selected date is not an NSE trading day. Please select another date."
            }
        )

    # 4. Count future trading days
    future_offset = count_future_trading_days(selected_date)
    if future_offset <= 0:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Selected date is not an NSE trading day. Please select another date."}
        )

    model_entry = MODELS[code]
    target_day_num = model_entry["last_day_num"] + future_offset

    # 5. Predict price using scikit-learn LinearRegression model
    pred_val = model_entry["model"].predict([[target_day_num]])[0]
    pred_rounded = round(float(pred_val), 2)

    company_info = COMPANIES[code]

    return {
        "success": True,
        "company_id": code,
        "company_name": company_info["name"],
        "symbol": company_info["symbol"],
        "selected_date": selected_date.strftime("%Y-%m-%d"),
        "formatted_date": selected_date.strftime("%d-%m-%Y"),
        "display_date": selected_date.strftime("%d %B %Y"),
        "day_name": selected_date.strftime("%A"),
        "day_number": target_day_num,
        "predicted_price": pred_rounded,
        "predicted_price_str": f"₹ {pred_rounded:.2f}"
    }

@app.get("/", response_class=HTMLResponse)
def get_index():
    return HTML_PAGE

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stock Price Prediction - Linear Regression</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background-color: #f7f9fa;
    color: #1a1a1a;
    display: flex;
    justify-content: center;
    padding: 24px 16px 40px 16px;
  }
  .container {
    width: 100%;
    max-width: 820px;
    background: #ffffff;
    border: 1px solid #dcdfe4;
    border-radius: 8px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    padding: 34px 38px;
  }
  .header {
    text-align: center;
    border-bottom: 2px solid #0056b3;
    padding-bottom: 16px;
    margin-bottom: 24px;
  }
  .header h1 {
    font-size: 23px;
    font-weight: 700;
    color: #003366;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }
  .header h2 {
    font-size: 16px;
    font-weight: 500;
    color: #555555;
    margin-top: 5px;
  }
  .header .data-source {
    font-size: 13px;
    color: #285f8f;
    font-weight: 600;
    margin-top: 7px;
  }
  .section-divider {
    border: none;
    border-top: 1px solid #e5e8ec;
    margin: 22px 0;
  }
  /* Input Section */
  .form-group {
    margin-bottom: 16px;
    text-align: center;
  }
  .form-label {
    font-size: 14.5px;
    font-weight: 600;
    color: #333333;
    display: block;
    margin-bottom: 8px;
  }
  .form-control {
    font-size: 15px;
    padding: 9px 14px;
    border: 1px solid #bcc3ca;
    border-radius: 6px;
    outline: none;
    background: #ffffff;
    color: #222222;
    min-width: 260px;
    cursor: pointer;
  }
  .form-control:focus {
    border-color: #0056b3;
    box-shadow: 0 0 0 3px rgba(0, 86, 179, 0.15);
  }
  .btn-predict {
    background-color: #0056b3;
    color: #ffffff;
    font-size: 15px;
    font-weight: 600;
    padding: 10px 32px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    letter-spacing: 0.6px;
    margin-top: 6px;
    transition: background-color 0.15s ease;
  }
  .btn-predict:hover {
    background-color: #003d82;
  }
  .error-box {
    color: #b02a37;
    background-color: #f8d7da;
    border: 1px solid #f5c2c7;
    padding: 10px 16px;
    border-radius: 6px;
    font-size: 14px;
    text-align: center;
    margin-top: 14px;
    display: none;
  }
  /* Prediction Output Card */
  .prediction-card {
    text-align: center;
    background-color: #f0f7ff;
    border: 1px solid #b8daff;
    border-radius: 8px;
    padding: 22px 18px;
    margin: 18px 0;
  }
  .pred-title {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
    color: #004085;
    text-transform: uppercase;
  }
  .pred-price {
    font-size: 38px;
    font-weight: 700;
    color: #002752;
    margin: 8px 0;
    font-family: 'Consolas', 'Courier New', monospace;
  }
  .pred-company {
    font-size: 16px;
    font-weight: 600;
    color: #1a395a;
  }
  .pred-date {
    font-size: 14.5px;
    color: #495057;
    margin-top: 3px;
  }
  /* Graph Section */
  .graph-title {
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    text-align: center;
    color: #495057;
    margin-bottom: 12px;
  }
  .chart-wrapper {
    position: relative;
    width: 100%;
    height: 380px;
  }
  /* Bottom Disclaimer */
  .disclaimer {
    margin-top: 26px;
    text-align: center;
    font-size: 12.5px;
    color: #6c757d;
    line-height: 1.5;
    border-top: 1px solid #e9ecef;
    padding-top: 16px;
  }
</style>
</head>
<body>

<div class="container">
  <!-- Header -->
  <div class="header">
    <h1>Stock Price Prediction</h1>
    <h2>Linear Regression</h2>
    <div class="data-source" id="dataSourceText">Historical NSE Data (FY 2025–26)</div>
  </div>

  <!-- Form Controls -->
  <div class="form-group">
    <label class="form-label" for="companySelect">Select Company:</label>
    <select class="form-control" id="companySelect" onchange="handleCompanyChange()">
      <option value="SBIN">State Bank of India</option>
      <option value="RELIANCE">Reliance Industries</option>
      <option value="TCS">Tata Consultancy Services</option>
      <option value="INFY">Infosys</option>
      <option value="HDFCBANK">HDFC Bank</option>
      <option value="ICICIBANK">ICICI Bank</option>
      <option value="ITC">ITC</option>
      <option value="LT">Larsen & Toubro</option>
      <option value="BHARTIARTL">Bharti Airtel</option>
      <option value="TATAMOTORS">Tata Motors</option>
    </select>
  </div>

  <div class="form-group">
    <label class="form-label" for="targetDate">Select Future Date:</label>
    <input type="date" class="form-control" id="targetDate" min="2026-04-01" value="2026-04-15">
  </div>

  <div style="text-align: center;">
    <button class="btn-predict" onclick="handlePredict()">PREDICT</button>
  </div>

  <div class="error-box" id="errorBox"></div>

  <hr class="section-divider">

  <!-- Prediction Display -->
  <div class="prediction-card" id="predCard">
    <div class="pred-title">PREDICTED STOCK PRICE</div>
    <div class="pred-price" id="predPriceText">₹ 1103.80</div>
    <div class="pred-company" id="predCompanyText">State Bank of India</div>
    <div class="pred-date" id="predDateText">15 April 2026</div>
  </div>

  <hr class="section-divider">

  <!-- Graph -->
  <div class="graph-section">
    <div class="graph-title">GRAPH</div>
    <div class="chart-wrapper">
      <canvas id="stockChart"></canvas>
    </div>
  </div>

  <!-- Disclaimer -->
  <div class="disclaimer">
    Prediction is based on historical stock data and Linear Regression. It is an estimate and is not guaranteed to be the actual future market price.
  </div>
</div>

<script>
let chartInstance = null;
let currentCompanyData = null;

async function loadCompanyData(companyId) {
  try {
    const res = await fetch(`/api/company_data?company=${companyId}`);
    currentCompanyData = await res.json();

    document.getElementById('dataSourceText').textContent =
      `Historical NSE Data: ${currentCompanyData.name} (${currentCompanyData.symbol})`;

    document.getElementById('targetDate').min = currentCompanyData.min_future_date;

    renderChart();
    await handlePredict();
  } catch (err) {
    console.error("Failed to load company data:", err);
  }
}

function renderChart() {
  const ctx = document.getElementById('stockChart').getContext('2d');

  if (chartInstance) {
    chartInstance.destroy();
  }

  chartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [...currentCompanyData.dates],
      datasets: [
        {
          label: 'Historical Closing Price',
          data: [...currentCompanyData.actual_prices],
          borderColor: '#1f77b4',
          backgroundColor: 'transparent',
          borderWidth: 1.8,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.1
        },
        {
          label: 'Linear Regression Trend Line',
          data: [...currentCompanyData.trend_prices],
          borderColor: '#2ca02c',
          backgroundColor: 'transparent',
          borderWidth: 2,
          borderDash: [5, 4],
          pointRadius: 0,
          pointHoverRadius: 0
        },
        {
          label: 'Future Predicted Point',
          data: [],
          borderColor: '#d62728',
          backgroundColor: '#d62728',
          pointRadius: 6,
          pointHoverRadius: 8,
          showLine: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false,
      },
      plugins: {
        title: {
          display: true,
          text: `${currentCompanyData.name} Stock Price Prediction`,
          font: { size: 14, weight: 'bold' },
          color: '#212529',
          padding: { bottom: 10 }
        },
        legend: {
          position: 'top',
          labels: {
            font: { size: 12 },
            boxWidth: 16
          }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return context.dataset.label + ': ₹ ' + context.parsed.y.toFixed(2);
            }
          }
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Date',
            font: { size: 12, weight: 'bold' }
          },
          ticks: {
            maxTicksLimit: 12,
            font: { size: 11 }
          },
          grid: { color: '#f0f0f0' }
        },
        y: {
          title: {
            display: true,
            text: 'Closing Price (₹)',
            font: { size: 12, weight: 'bold' }
          },
          ticks: {
            font: { size: 11 },
            callback: function(val) { return '₹ ' + val; }
          },
          grid: { color: '#e9ecef' }
        }
      }
    }
  });
}

async function handleCompanyChange() {
  const companyId = document.getElementById('companySelect').value;
  await loadCompanyData(companyId);
}

async function handlePredict() {
  const companyId = document.getElementById('companySelect').value;
  const dateInput = document.getElementById('targetDate').value;
  const errorBox = document.getElementById('errorBox');
  const predPriceText = document.getElementById('predPriceText');
  const predCompanyText = document.getElementById('predCompanyText');
  const predDateText = document.getElementById('predDateText');

  errorBox.style.display = 'none';

  if (!dateInput) {
    errorBox.textContent = "Please select a future date.";
    errorBox.style.display = 'block';
    return;
  }

  try {
    const res = await fetch(`/api/predict?company=${companyId}&date=${dateInput}`);
    const data = await res.json();

    if (!res.ok || !data.success) {
      errorBox.textContent = data.error || "Selected date is not an NSE trading day. Please select another date.";
      errorBox.style.display = 'block';
      return;
    }

    // Update text outputs
    predPriceText.textContent = data.predicted_price_str;
    predCompanyText.textContent = data.company_name;
    predDateText.textContent = data.display_date;

    // Update graph
    updateChart(data.formatted_date, data.predicted_price, data.company_name);

  } catch (err) {
    errorBox.textContent = "Error communicating with server.";
    errorBox.style.display = 'block';
  }
}

function updateChart(futureDateStr, predictedPrice, companyName) {
  if (!chartInstance || !currentCompanyData) return;

  chartInstance.options.plugins.title.text = `${companyName} Stock Price Prediction`;

  // New labels with the future date
  const newLabels = [...currentCompanyData.dates, futureDateStr];
  chartInstance.data.labels = newLabels;

  // Extend historical with null
  chartInstance.data.datasets[0].data = [...currentCompanyData.actual_prices, null];

  // Extend trend line to predicted price
  chartInstance.data.datasets[1].data = [...currentCompanyData.trend_prices, predictedPrice];

  // Future predicted point: nulls + single predicted point at end
  const futurePoints = new Array(currentCompanyData.dates.length).fill(null);
  futurePoints.push(predictedPrice);
  chartInstance.data.datasets[2].data = futurePoints;

  chartInstance.update();
}

window.addEventListener('DOMContentLoaded', () => {
  loadCompanyData('SBIN');
});
</script>

</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
