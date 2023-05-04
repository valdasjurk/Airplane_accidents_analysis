import requests
import json

""" Prints airport full name by its iata code"""


def add_data_from_airlabs_api():
    params = {
        "api_key": "ef006315-1a52-4296-8ba9-124b6fb58f67",
        "iata_code": ["KUN, JAX, EWR, VNO"],
    }
    method = "ping"
    api_base = "https://airlabs.co/api/v9/airports?"
    api_result = requests.get(api_base + method, params)
    data = api_result.text
    data_dict = json.loads(data)

    for key in data_dict["response"]:
        print(key["name"])


if __name__ == "__main__":
    add_data_from_airlabs_api()
