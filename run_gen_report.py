
from src.utils.helpers import return_full_path
from src.services.report.make_report import make_report

if __name__ == '__main__':
    ml_file_paths = [
        return_full_path('new_experiment/meta_results/python/codexglue/ml_codexglue_rougel_f.json'),
        return_full_path('new_experiment/meta_results/java/codexglue/ml_codexglue_rougel_f.json'),
        return_full_path('new_experiment/meta_results/python/wanetal/ml_wanetal_rougel_f.json'),
        return_full_path('new_experiment/meta_results/java/huetal/ml_huetal_rougel_f.json'),
    ]

    for path in ml_file_paths:
        make_report(path)