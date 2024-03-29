import json
import os

from src.meta_model.meta_utils import read_data, extract_features
from src.utils.helpers import return_full_path


if __name__ == '__main__':

    # lang = 'java'
    langs = ['python', 'java']

    # corpus_name = 'huetal'
    # corpus_name = 'codexglue'
    corpus_names = ['wanetal', 'huetal', 'codexglue']

    train_corpus_name = None

    for lang in langs:
        print(f'Lang: {lang}')
        for corpus_name in corpus_names:
            print(f'corpus_name: {corpus_name}')
            if lang == 'java':
                if corpus_name == 'wanetal': continue
                if corpus_name == 'huetal':
                    train_corpus_name = 'codexglue'
                else:
                    train_corpus_name = 'huetal'
            elif lang == 'python':
                # if corpus_name in ['huetal', 'wanetal']: continue
                if corpus_name == 'huetal': continue
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

            json_path = os.path.join(features_dir, f'{corpus_name}_features.json')

            with open(json_path, 'w') as file:
                json.dump(test_data, file, indent=4)
