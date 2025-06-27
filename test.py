# test_request.py
import requests

url = "http://127.0.0.1:5000/predict"
data = {"review": "This movie was terrible and boring."}

response = requests.post(url, json=data)
print(response.json())
