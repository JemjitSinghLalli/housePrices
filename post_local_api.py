import json
import requests

# Test Local
url = "http://127.0.0.1:8099/housePrices"

with open(
    "C:/Users/lallij/PycharmProjects/housePrices/data/example_data.json", "r"
) as f:
    payload = json.load(f)

resp = requests.post(url, data=json.dumps(payload, indent=4))
assert resp.status_code == 200
print(resp.text)
