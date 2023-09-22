def get_better_system_by_aval(sorted_measure: str, dict_systems: list[dict]):
    sorted_list = sorted(dict_systems, key = lambda x: x['measures'][sorted_measure], reverse = True)

    return sorted_list[0]['name']

def get_better_result_and_features(dict_example: list[dict], sorted_measure: str):
    systems = dict_example['systems']
    features = dict_example['features']
    better_system = get_better_system_by_aval(sorted_measure, systems)

    return [features['tokens_type'], better_system]