import json
import os

from src.meta_model.meta_utils import read_data, extract_features
from src.utils.helpers import return_full_path


if __name__ == '__main__':

    # lang = 'java'
    lang = 'python'

    # corpus_name = 'huetal'
    # corpus_name = 'codexglue'
    corpus_name = 'wanetal'

    eval_measure = 'rougel_f'
    # eval_measure = 'meteor_score'
    # eval_measure = 'bleu_4_o'

    train_corpus_name = None

    if lang == 'java':
        if corpus_name == 'huetal':
            train_corpus_name = 'codexglue'
        else:
            train_corpus_name = 'huetal'
    elif lang == 'python':
        if corpus_name == 'wanetal':
            train_corpus_name = 'codexglue'
        else:
            train_corpus_name = 'wanetal'
    else:
        print('ERRO "lang option" INVALID')
        exit(-1)

    test_path = return_full_path(f'new_experiment/descriptions_json/{lang}/{corpus_name}/'
                                 f'{corpus_name}.json')
    train_path = return_full_path(f'new_experiment/descriptions_json/{lang}/{train_corpus_name}/'
                                  f'{train_corpus_name}.json')

    test_data = read_data(test_path)
    train_data = read_data(train_path)

    print(f'\nTest Corpus: {corpus_name} - {len(test_data)}')
    print(f'\nTrain Corpus: {train_corpus_name} - {len(train_data)}\n')

    features_dir = return_full_path(f'new_experiment/features_files/{lang}/{corpus_name}')

    os.makedirs(features_dir, exist_ok=True)

    extract_features(test_data, train_data, max_desc_len=20)

    json_path = os.path.join(features_dir, f'{corpus_name}_{eval_measure}_features.json')

    with open(json_path, 'w') as file:
        json.dump(test_data, file, indent=4)
