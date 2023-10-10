import os
import time
from src.services.description.generate_description import generate_description
from src.services.description.get_better_result import get_all_better_results

if __name__ == '__main__':

    # lang, test_mode, dataset
    lst_parameters = [
        ('java', False, 'codexglue'),
        ('java', False, 'huetal'),
        ('python', False, 'codexglue'),
        ('python', False, 'wanetal'),
    ]

    print('---START---')
    for i in range(len(lst_parameters) - 1):
        print(f'Current parameters: {i}')
        start_time = time.time()
        print(lst_parameters[i][0])
        generate_description(lst_parameters[i][0], lst_parameters[i][1], lst_parameters[i][2])
        print(f'{i}: {time.time() - start_time}')
    
    print('---END---')

    """
    O print logo abaixo é para printar a tupla do experimento
    """
    
    # print(get_all_better_results(json_path='./descriptions_json/java/codexglue/codexglue.json'))


