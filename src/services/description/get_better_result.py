import json
import os


def read_json(file_path: str):
    with open(file_path, 'r') as json_file:
        return json.load(json_file)


def get_better_system_by_aval(sorted_measure: str, dict_systems: list[dict]):
    sorted_list = sorted(dict_systems, key=lambda x: x['measures'][sorted_measure], reverse=True)
    return sorted_list[0]['name']


def get_better_result_and_features(dict_example: dict, sorted_measure: str = 'meteor_score'):
    systems = dict_example['systems']
    systems = [s for s in systems if '_rougel_ft' in s['name']]
    features = dict_example['features']
    better_system = get_better_system_by_aval(sorted_measure, systems)
    return features['tokens_type'], better_system


def get_all_better_results(json_path: str, sorted_measure: str = 'meteor_score'):
    dicts = read_json(os.path.join(json_path))
    all_better_results = []
    for elem in dicts:
        t = get_better_result_and_features(elem, sorted_measure)
        all_better_results.append(t)
    return all_better_results
