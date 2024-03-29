import numpy as np
import os
import sys
import json

from src.utils import utils
from src.evaluation_measures.evaluation_measures import compute_rouge, compute_bleu, compute_meteor
from tqdm import tqdm
from src.evaluation_measures.bleu import compute_maps, bleu_from_maps
from src.utils.helpers import return_full_path


if __name__ == '__main__':


    langs = ['python', 'java']

    corpus_names = ['wanetal', 'huetal', 'codexglue']    

    for lang in langs:
        print(f'Lang: {lang}')
        for corpus_name in corpus_names:
            print(f'corpus_name: {corpus_name}')
            if lang == 'java' and corpus_name == 'wanetal': continue
            if lang == 'python' and corpus_name == 'huetal': continue

            preproc_config = 'none'

            systems_dir = return_full_path(f'new_experiment/meta_descriptions/{lang}/{corpus_name}')
            results_dir = return_full_path(f'new_experiment/meta_results/{lang}/{corpus_name}')

            test_file_path = return_full_path(f'new_experiment/corpora/{lang}/{corpus_name}/csv/'
                                            f'test_{preproc_config}.csv')

            size_threshold = -1

            max_desc_len = 20

            _, _, test_data = utils.read_corpus_csv(test_file_path=test_file_path, sample_size=size_threshold)

            os.makedirs(results_dir, exist_ok=True)

            test_descs = test_data[1]

            print(f'\nCorpus: {corpus_name} - {lang}')

            print(f'\n  Test set: {len(test_descs)}')

            dict_sys_descs = utils.read_descriptions(systems_dir)

            all_results = {}

            for cont, (sys_name, sys_descs) in enumerate(dict_sys_descs.items(), start=1):

                print(f'\nSystem {cont} / {len(dict_sys_descs)}: {sys_name} - {len(sys_descs)}')

                assert len(test_descs) == len(sys_descs)

                system_results = {}

                print()

                with tqdm(total=len(test_descs), file=sys.stdout, colour='blue',
                        desc='  Evaluating ') as pbar:

                    for sys_desc, ref_sys in zip(sys_descs, test_descs):

                        tokens_desc = sys_desc.split(' ')

                        if len(tokens_desc) > max_desc_len:
                            tokens_desc = tokens_desc[:max_desc_len]
                            sys_desc = ' '.join(tokens_desc).strip()

                        rouge_scores = compute_rouge(ref_sys, sys_desc, max_desc_len)

                        bleu_score = compute_bleu(ref_sys, sys_desc)

                        meteor_score = compute_meteor(ref_sys.split(' '), tokens_desc)

                        for rouge_n, metrics in rouge_scores.items():
                            for metric, value in metrics.items():
                                metric_name = rouge_n.replace('-', '') + '_' + metric
                                if metric_name in system_results:
                                    system_results[metric_name].append(value)
                                else:
                                    system_results[metric_name] = [value]

                        if 'meteor_score' in system_results:
                            system_results['meteor_score'].append(meteor_score)
                        else:
                            system_results['meteor_score'] = [meteor_score]

                        if 'bleu_4' in system_results:
                            system_results['bleu_4'].append(bleu_score)
                        else:
                            system_results['bleu_4'] = [bleu_score]

                        gold_map, prediction_map = compute_maps([sys_desc], [ref_sys])

                        bleu_scores = bleu_from_maps(gold_map, prediction_map)

                        bleu_4_o = bleu_scores[0] / 100

                        if 'bleu_4_o' in system_results:
                            system_results['bleu_4_o'].append(bleu_4_o)
                        else:
                            system_results['bleu_4_o'] = [bleu_4_o]

                        pbar.update(1)

                    all_results[sys_name] = system_results

            metrics_columns = ['rouge1_r', 'rouge1_p', 'rouge1_f', 'rouge2_r', 'rouge2_p', 'rouge2_f',
                            'rougel_r', 'rougel_p', 'rougel_f', 'meteor_score', 'bleu_4', 'bleu_4_o']

            report = 'system'

            for metric in metrics_columns:
                report += ';mean_' + metric + ';' 'std_' + metric

            for system_name, results in all_results.items():

                report += '\n' + system_name

                dict_system_results = {}

                for metric in metrics_columns:

                    values = results[metric]

                    report += ';' + str(np.mean(values)).replace('.', ',') + ';' + \
                            str(np.std(values)).replace('.', ',')

            json_path = os.path.join(results_dir, f'{corpus_name}_results.json')

            with open(json_path, 'w') as file:
                json.dump(all_results, file, indent=4)

            report_file = os.path.join(results_dir,  f'{corpus_name}_results_report.csv')

            with open(report_file, 'w') as file:
                file.write(report)