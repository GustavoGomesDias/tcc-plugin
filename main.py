import time
import os

from src.services.description.generate_description import generate_description


if __name__ == '__main__':

    is_turn_off_computer = False

    # lang, test_mode, dataset

    lst_parameters = [
        ('java', False, 'codexglue'),
        ('java', False, 'huetal'),
        ('python', False, 'codexglue'),
        ('python', False, 'wanetal'),
    ]

    print('\n---START---\n')

    for i in range(len(lst_parameters)):

        print(f'\nCurrent parameters: {i} -- {lst_parameters[i][0]}')

        start_time = time.time()

        generate_description(lst_parameters[i][0], lst_parameters[i][1], lst_parameters[i][2])

        print(f'\n{i}: {time.time() - start_time}')
    
    print('\n---END---\n')

    if is_turn_off_computer:

        print('\n\nTurning off computer ....')

        time.sleep(60)

        os.system('shutdown -h now')
