import json

import config
import requests


def add_data_from_airlabs_api(api_key=config.AIRLABS_API_KEY):
    """Prints airport full name by its iata code"""
    params = {
        "api_key": api_key,
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
