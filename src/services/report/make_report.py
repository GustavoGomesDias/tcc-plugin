from src.services.report.read_data import read_json
import csv
from src.utils.helpers import return_full_path

def make_report(file_path: str):
    data = read_json(file_path)

    with open(return_full_path('new_experiment/reports/regression_result.csv'), 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Algoritmo', 'RMSE', 'MAE', 'Person'])
        for reg_name in data.keys():
            rmse = data[reg_name]['rmse']
            mae = data[reg_name]['mae']
            person = data[reg_name]['pearson_corr']

            writer.writerow([reg_name, rmse, mae, person])
