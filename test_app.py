import sys
import urllib.request
import urllib.error
import json

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

base_url = "http://127.0.0.1:5000"

print("--- Testing API Data Endpoint ---")
with urllib.request.urlopen(f"{base_url}/api/data") as r:
    data = json.loads(r.read().decode())
    print("Historical data points loaded:", len(data["dates"]))
    print("Start Date:", data["dates"][0], "Actual Close: ₹", data["actual_prices"][0])
    print("End Date  :", data["dates"][-1], "Actual Close: ₹", data["actual_prices"][-1])

print("\n--- Testing Future Date Predictions ---")
test_dates = [
    "2026-04-10",  # Trading Day 7 after March 31
    "2026-05-15",  # Trading Day 30 after March 31
    "2026-06-01",  # Trading Day 41 after March 31
    "2026-04-05",  # Sunday (Weekend)
    "2026-04-03",  # Good Friday (NSE Holiday)
]

for d in test_dates:
    url = f"{base_url}/api/predict?date={d}"
    try:
        with urllib.request.urlopen(url) as r:
            res = json.loads(r.read().decode())
            print(f"Date {d} ({res['day_name']}):")
            print(f"  -> Day Number      : {res['day_number']}")
            print(f"  -> Predicted Price : {res['predicted_price_str']}")
            print(f"  -> Formatted Date  : {res['formatted_date']}")
    except urllib.error.HTTPError as e:
        err = json.loads(e.read().decode())
        print(f"Date {d}:")
        print(f"  -> Status {e.code}: {err['error']}")
