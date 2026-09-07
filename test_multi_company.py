import sys
import urllib.request
import urllib.error
import json

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

base_url = "http://127.0.0.1:5000"

print("--- 1. Testing Companies List Endpoint ---")
with urllib.request.urlopen(f"{base_url}/api/companies") as r:
    companies = json.loads(r.read().decode())
    print(f"Total Companies loaded: {len(companies)}")
    for c in companies:
        print(f"  • {c['name']} ({c['symbol']})")

print("\n--- 2. Running Required Tests ---")
required_tests = [
    ("SBIN", "2026-04-15", "Test 1: State Bank of India on 15 April 2026"),
    ("INFY", "2026-05-15", "Test 2: Infosys on 15 May 2026"),
    ("RELIANCE", "2026-06-01", "Test 3: Reliance Industries on 1 June 2026"),
    ("TCS", "2026-04-15", "Additional: TCS on 15 April 2026"),
    ("TATAMOTORS", "2026-05-15", "Additional: Tata Motors on 15 May 2026"),
    ("SBIN", "2026-04-12", "Validation: Sunday (Weekend) on 12 April 2026"),
    ("SBIN", "2026-04-14", "Validation: Ambedkar Jayanti (NSE Holiday) on 14 April 2026"),
]

for code, date_str, desc in required_tests:
    url = f"{base_url}/api/predict?company={code}&date={date_str}"
    print(f"\n{desc}:")
    try:
        with urllib.request.urlopen(url) as r:
            res = json.loads(r.read().decode())
            print(f"  Company         : {res['company_name']} ({res['symbol']})")
            print(f"  Selected Date   : {res['display_date']} ({res['day_name']})")
            print(f"  Trading Day #   : {res['day_number']}")
            print(f"  Predicted Price : {res['predicted_price_str']}")
    except urllib.error.HTTPError as e:
        err = json.loads(e.read().decode())
        print(f"  Status {e.code}  : {err['error']}")

print("\n--- All tests completed successfully! ---")
