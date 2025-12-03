import requests
import json
import os

url = "http://localhost:5000/api/predict"
payload = {
    "filename": "chatBoys.txt",
    "selected_user": "Saim SE"
}
headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        print("Prediction successful!")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Prediction failed with status code {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Error: {e}")
