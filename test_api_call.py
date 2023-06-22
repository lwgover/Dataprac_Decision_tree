import requests
query = {'independent_variable':'DP54', 'dependent_variables':'DP1,DP4'}
response = requests.get('http://127.0.0.1:5000/', params=query)
print(response.json())
