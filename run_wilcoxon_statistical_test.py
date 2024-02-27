import scipy.stats as stats
import json
import numpy as np

from src.utils.helpers import return_full_path

"""
    https://www.geeksforgeeks.org/how-to-conduct-a-wilcoxon-signed-rank-test-in-python/
"""

if __name__ == '__main__':

    lang = 'java'
    # lang = 'python'

    corpus_name = 'huetal'
    # corpus_name = 'codexglue'
    # corpus_name = 'wanetal'

    eval_measure = 'rougel_f'
    # eval_measure = 'bleu_4_o'

    results_file = return_full_path(f'new_experiment/meta_results/{lang}/{corpus_name}/'
                                    f'{corpus_name}_results.json')

    with open(results_file) as data_file:
        systems_results_dict = json.load(data_file)

    for system_name, system_results in systems_results_dict.items():

        system_results_eval_measure = system_results[eval_measure]

        print(f'\nSystem: {system_name} -- {np.mean(system_results_eval_measure)}\n')

        for other_system_name, other_system_results in systems_results_dict.items():

            if system_name != other_system_name:

                other_system_results_eval_measure = other_system_results[eval_measure]

                statistical_test = stats.wilcoxon(system_results_eval_measure,
                                                  other_system_results_eval_measure)

                print(f'\tSystem 2: {other_system_name} -- {np.mean(other_system_results_eval_measure)}')

                print(f'\t\tStatistical Test Results: {statistical_test.pvalue:.6f}')

    """
        Gerar um CSV com os resultados os testes estatísticos cada sistema vs os outros sistemas.
        
        	A	     B	      C
        A	1	   0.001	0.02
        B	0.001	1	0.07
        C	0.02	0.07	1
        
        os valores das células são os statistical_test.pvalue
    """

