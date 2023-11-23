import json

def read_json(file_path: str) -> dict:
    file = open(file_path)

    json_data = json.load(file)
    
    return json_data