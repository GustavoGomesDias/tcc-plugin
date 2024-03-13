import os
import scipy.stats as stats
import json
import numpy as np
import pandas as pd

from src.utils.helpers import return_full_path

"""
    https://www.geeksforgeeks.org/how-to-conduct-a-wilcoxon-signed-rank-test-in-python/
"""

if __name__ == '__main__':

    langs = ['python', 'java']

    corpus_names = ['wanetal', 'huetal', 'codexglue']

    regressions = ['linear_svm_bleu_4_o', 'linear_regression_bleu_4_o', 'xgbregressor_bleu_4_o', 'lgbmregressor_bleu_4_o', 'catboostregressor_bleu_4_o', 'svr_bleu_4_o', 'bayesianridge_bleu_4_o', 'decision_tree_regressor_bleu_4_o']

    dict_results = dict()

    for lang in langs:
        print(f'Lang: {lang}')
        for corpus_name in corpus_names:
            print(f'corpus_name: {corpus_name}')
            if lang == 'java' and corpus_name == 'wanetal': continue
            if lang == 'python' and corpus_name == 'huetal': continue

            eval_measure = 'rougel_f'
            # eval_measure = 'bleu_4_o'

            dict_results[f'{lang}_{corpus_name}'] = {}

            results_file = return_full_path(f'new_experiment/meta_results/{lang}/{corpus_name}/'
                                            f'{corpus_name}_results.json')

            with open(results_file) as data_file:
                systems_results_dict = json.load(data_file)

            for system_name, system_results in systems_results_dict.items():

                system_results_eval_measure = system_results[eval_measure]

                # print(f'\nSystem: {system_name} -- {np.mean(system_results_eval_measure)}\n')

                if system_name in regressions: continue 
                dict_results[f'{lang}_{corpus_name}'][system_name] = {}

                for other_system_name, other_system_results in systems_results_dict.items():

                    if system_name != other_system_name:

                        other_system_results_eval_measure = other_system_results[eval_measure]

                        statistical_test = stats.wilcoxon(system_results_eval_measure,
                                                        other_system_results_eval_measure)

                        # print(f'\tSystem 2: {other_system_name} -- {np.mean(other_system_results_eval_measure)}')

                        dict_results[f'{lang}_{corpus_name}'][system_name][other_system_name] = f'{statistical_test.pvalue:.6f}'
                        # print(f'\t\tStatistical Test Results: {statistical_test.pvalue:.6f}')
    # print(dict_results)
    with pd.ExcelWriter('statistical_test.xlsx', engine='xlsxwriter', mode='w') as writer:                   
        excel_file_path = 'statistical_test.xlsx'

        if not os.path.exists(excel_file_path):
            # If it doesn't exist, create a new Excel file
            writer = pd.ExcelWriter(excel_file_path, engine='openpyxl')
            writer.close()
        sheets = list(dict_results.keys())
        # print(dict_results)
        for sheet in sheets:
            lines = []
            headers = dict_results[sheet].keys()
            models =  dict_results[sheet].keys()
            for header in headers:
                line = [header]
                for model in models:
                    if header == model:
                        line.append(1)
                    else:
                        line.append(float(dict_results[sheet][header][model]))
                lines.append(line)

            df = pd.DataFrame(lines, columns=['systems'] + list(headers))
            df.to_excel(writer, index=False, sheet_name=sheet)

"""
    Gerar um CSV com os resultados os testes estatísticos cada sistema vs os outros sistemas.
    
        A	     B	      C
    A	1	   0.001	0.02
    B	0.001	1	0.07
    C	0.02	0.07	1
    
    os valores das células são os statistical_test.pvalue
"""

