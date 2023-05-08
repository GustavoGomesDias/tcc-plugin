import pandas as pd
import os


def read_corpus_csv(train_file_path=None, valid_file_path=None, test_file_path=None, sample_size=-1):
    train_data = read_csv_file(train_file_path, sample_size)
    valid_data = read_csv_file(valid_file_path, sample_size)
    test_data = read_csv_file(test_file_path, sample_size)
    return train_data, valid_data, test_data


def read_csv_file(file_path, sample_size=-1):
    if file_path is None:
        return None
    assert os.path.exists(file_path), f'Error. File {file_path} does not exist.'
    if sample_size > 0:
        df = pd.read_csv(file_path, sep='\t', na_filter=False, nrows=sample_size)
    else:
        df = pd.read_csv(file_path, sep='\t', na_filter=False)
    tokens = df['code'].tolist()
    descriptions = df['desc'].tolist()
    tokens_original = None
    if 'code_original' in df:
        tokens_original = df['code_original'].tolist()
    return tokens, descriptions, tokens_original


def read_descriptions(systems_dir):
    assert os.path.exists(systems_dir), f'Error. Directory {systems_dir} does not exist.'
    systems_names = os.listdir(systems_dir)
    sys_descriptions = {}
    for system_name in systems_names:
        system_file = os.path.join(systems_dir, system_name)
        with open(system_file) as file:
            content = file.read()
        lines = content.split('\n')
        if '.' in system_name:
            system_name = system_name.split('.')[0]
        sys_descriptions[system_name] = lines
    return sys_descriptions
