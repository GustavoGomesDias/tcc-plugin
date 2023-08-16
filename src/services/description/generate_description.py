import os
import json
import sys
import logging

from src.evaluation_measures.evaluation_measures import compute_rouge, compute_bleu, compute_meteor, compute_bert_score
from src.evaluation_measures.bleu import compute_maps, bleu_from_maps
from src.utils import corpora
from tqdm import tqdm
from src.utils.helpers import return_full_path
from src.services.tokens.Tokenize import Tokenize
from src.utils.types.Language import Language


logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)

# TODO: talvez seja necessário ajeitar os paths nas linhas 32, 34 e 40

def generate_description(lang, test_mode = False):
    # corpus_name = 'huetal'
    corpus_name = 'codexglue'
    # corpus_name = 'wanetal'

    preproc_config = 'none'

    if lang == 'java':
        preproc_config = 'camelsnakecase'
        language = Language.JAVA
    else:
        language = Language.PYTHON

    systems_dir = return_full_path(f'descriptions/{lang}/{corpus_name}')

    json_desc_dir = return_full_path(f'descriptions_json/{lang}/{corpus_name}')

    size_threshold = -1

    max_desc_len = 20

    test_file_path = return_full_path(f'corpora/{lang}/{corpus_name}/test_{preproc_config}.csv')

    _, _, test_data = corpora.read_corpus_csv(test_file_path=test_file_path, sample_size=size_threshold)

    os.makedirs(json_desc_dir, exist_ok=True)

    sys_descriptions = corpora.read_descriptions(systems_dir)

    codes = test_data[0]
    descriptions = test_data[1]

    print(f'\nCorpus: {corpus_name} - {len(codes)} - {len(descriptions)} - {preproc_config}')

    print(f'\nTotal of Systems: {len(sys_descriptions)}\n')

    data = []
    tokenize_service = Tokenize(language)

    # with tqdm(total=len(codes), file=sys.stdout, colour='blue', desc='  Evaluating ') as pbar:

    for i, (code, ref_desc) in enumerate(zip(codes, descriptions)):

        features = tokenize_service.count_token_type([code])

        dict_example = {
            'id': i + 1,
            'code': code,
            'ref_desc': ref_desc,
            'features': features
        }

        ref_desc_tokens = ref_desc.split(' ')

        systems_descs = []

        for sys_name in sys_descriptions.keys():

            sys_desc = sys_descriptions[sys_name][i]

            tokens_desc = sys_desc.split(' ')

            rouge_scores = compute_rouge(ref_desc, sys_desc, max_desc_len)

            bleu_4 = compute_bleu(ref_desc, sys_desc)

            meteor_score = compute_meteor(ref_desc_tokens, tokens_desc)

            P, R, F = compute_bert_score(ref_desc, sys_desc)

            measures = {}

            for rouge_n, metrics in rouge_scores.items():
                rouge_n = rouge_n.replace('-', '')
                for metric, value in metrics.items():
                    metric_name = f'{rouge_n}_{metric}'
                    measures[metric_name] = value

            measures['meteor_score'] = meteor_score
            measures['bleu_4'] = bleu_4
            measures['bert_score'] = {
                'P': float(P),
                'R': float(R),
                'F': float(F)
            }

            gold_map, prediction_map = compute_maps([sys_desc], [ref_desc])

            bleu_scores = bleu_from_maps(gold_map, prediction_map)

            measures['bleu_4_o'] = bleu_scores[0] / 100
            system_dict = {
                'name': sys_name,
                'description': sys_desc,
                'measures': measures,
            }

            systems_descs.append(system_dict)

        dict_example['systems'] = systems_descs
        data.append(dict_example)

        if test_mode and i == 5: break

        print(i)

        # pbar.update(1)

    json_path = os.path.join(json_desc_dir, corpus_name + '.json')
    with open(json_path, 'w') as file:
        json.dump(data, file)