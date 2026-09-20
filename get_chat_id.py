import requests

BOT_TOKEN = "8925514810:AAFvX0kddOiZQd0ndnmkHr54_0PSl2A3qQ0"

url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"

response = requests.get(url)

print(response.json())