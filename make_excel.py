import pandas as pd
import json
from src.utils.helpers import return_full_path
import os

root_path = return_full_path('new_experiment/meta_results')
excel_file_path = 'output.xlsx'

if not os.path.exists(excel_file_path):
    # If it doesn't exist, create a new Excel file
    writer = pd.ExcelWriter(excel_file_path, engine='openpyxl')
    writer.close()

with pd.ExcelWriter(excel_file_path, engine='xlsxwriter', mode='w') as writer:
    for folder_name, subfolders, files in os.walk(root_path):
        for file_name in files:
            if file_name.endswith('.csv'):
                csv_file_path = os.path.join(folder_name, file_name)

                path_splited = folder_name.split('/')
                sheet_name = file_name.split('.')[0]
                try:
                    df = pd.read_csv(csv_file_path, sep=';')
                    
                    df.to_excel(writer, sheet_name=f'{path_splited[-2]}-{sheet_name}', index=False, columns=['system', 'mean_rougel_f', 'std_rougel_f', 'mean_meteor_score', 'std_meteor_score', 'mean_bleu_4_o', 'std_bleu_4_o'])
                except Exception as e:
                    print(e)
    
regressions = ['java/codexglue/ml_java_codexglue_bleu_4_o.json', 'java/huetal/ml_java_huetal_bleu_4_o.json', 'python/codexglue/ml_python_codexglue_bleu_4_o.json', 'python/wanetal/ml_python_wanetal_bleu_4_o.json']

for regression in regressions:

    json_path = return_full_path('new_experiment/meta_results/java/codexglue/ml_java_codexglue_bleu_4_o.json')

    f = open(json_path)
    data: dict = json.load(f)


    headers = ['regressions', 'rmse', 'mae', 'r2', 'pearson_corr', 'kendall_tau']

    lines = []
    for key, value in data.items():
        line = [key]
        for header in headers:
            if header != 'regressions':
                if header in ['pearson_corr', 'kendall_tau']:
                    line.append(value[header][0])
                else:
                    line.append(value[header])
        lines.append(line)

    splited_regression = regression.split('/')
    tab_name = splited_regression[-1].split('.')[0]
    df = pd.DataFrame(lines, columns=headers)
    with pd.ExcelWriter('output.xlsx', engine='openpyxl', mode='a') as writer:
        df.to_excel(writer, index=False, sheet_name=tab_name)
