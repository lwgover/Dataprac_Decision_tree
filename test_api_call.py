import requests
response = requests.get('http://127.0.0.1:5000/tree/DP54,DP1,DP4')
print(response.json())
