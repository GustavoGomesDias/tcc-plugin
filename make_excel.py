import pandas as pd
from src.utils.helpers import return_full_path
import os

root_path = return_full_path('experiment_csv/results')
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

                sheet_name = file_name.split('.')[0]

                print(sheet_name)
                try:
                    df = pd.read_csv(csv_file_path, sep=';')
                    
                    df.to_excel(writer, sheet_name=sheet_name, index=False, columns=['system', 'mean_rougel_f', 'std_rougel_f', 'mean_meteor_score', 'std_meteor_score', 'mean_bleu_4_o', 'std_bleu_4_o'])
                except Exception as e:
                    print(e)