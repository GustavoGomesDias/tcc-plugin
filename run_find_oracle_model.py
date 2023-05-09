import json
import sys
import os

from tqdm import tqdm
from pathlib import Path
from helpers import return_full_path

if __name__ == '__main__':

    # lang = 'java'
    lang = 'python'

    # corpus_name = 'huetal'
    # corpus_name = 'codexglue'
    corpus_name = 'wanetal'

    measure_name = 'rougel_f'
    # measure_name = 'meteor_score'
    # measure_name = 'bleu_4_o'

    systems_descs_path = return_full_path(f'descriptions_json/{lang}/{corpus_name}/{corpus_name}.json')

    with open(systems_descs_path) as data_file:
        all_data = json.load(data_file)

    print('\nAll data:', len(all_data))

    oracle_value = 0.0

    print('\nEvaluating\n')

    with tqdm(total=len(all_data), colour='red', file=sys.stdout, desc='  Oracle: 0.0') as pbar:

        for i, example in enumerate(all_data):

            systems = example['systems']

            systems = sorted(systems, reverse=True,
                             key=lambda d: d['measures'][measure_name])

            oracle_value += systems[0]['measures'][measure_name]

            pbar.update(1)

            pbar.set_description('  Oracle: {:.2f}'.format(100 * (oracle_value / (i+1))))

    oracle_value /= len(all_data)

    print('\nOracle:', oracle_value * 100)
