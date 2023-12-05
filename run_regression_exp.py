import os
import json

from src.meta_model.meta_utils import read_data, build_regression_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from src.meta_model.meta_utils import evaluate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, kendalltau

from src.utils.helpers import return_full_path


if __name__ == '__main__':

    lang = 'java'
    # lang = 'python'

    corpus_name = 'huetal'
    # corpus_name = 'codexglue'
    # corpus_name = 'wanetal'

    eval_measure = 'rougel_f'
    # eval_measure = 'meteor_score'
    # eval_measure = 'bleu_4_o'

    train_corpus_name = None

    if lang == 'java':
        if corpus_name == 'huetal':
            train_corpus_name = 'codexglue'
        else:
            train_corpus_name = 'huetal'
    elif lang == 'python':
        if corpus_name == 'wanetal':
            train_corpus_name = 'codexglue'
        else:
            train_corpus_name = 'wanetal'
    else:
        print('ERROR "lang option" INVALID!')
        exit(-1)

    gen_desc_dir = return_full_path(f'new_experiment/meta_descriptions/{lang}/{corpus_name}')
    results_dir = return_full_path(f'new_experiment/meta_results/{lang}/{corpus_name}')

    os.makedirs(gen_desc_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    test_path = return_full_path(f'new_experiment/features_files/{lang}/{corpus_name}/{corpus_name}_features.json')
    train_path = return_full_path(f'new_experiment/features_files/{lang}/{train_corpus_name}/{train_corpus_name}_features.json')

    corpus_data = read_data(test_path)
    train_data = read_data(train_path)

    print(f'\nTest Corpus: {corpus_name} - {len(corpus_data)} - {eval_measure}')

    train_features, train_scores = build_regression_data(train_data, eval_measure)

    # train_features = train_features[:100]
    # train_scores = train_scores[:100]

    print(f'\nTrain Corpus: {train_corpus_name} - {len(train_data)} -- {len(train_features)}')

    scaler = MinMaxScaler()

    train_features = scaler.fit_transform(train_features)

    regressors_dict = {
        'Linear Regression': LinearRegression(n_jobs=-1),
        'BayesianRidge': BayesianRidge(),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
        'Linear SVM': LinearSVR(max_iter=1000, random_state=42),
        'SVR': SVR(),
        'LGBMRegressor': LGBMRegressor(random_state=42),
        'XGBRegressor':  XGBRegressor(n_estimators=100, n_jobs=-1, random_state=42),
        'CatBoostRegressor': CatBoostRegressor(verbose=0, n_estimators=100),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
        'MLP Regressor': MLPRegressor(max_iter=1000, random_state=42)
    }

    print(f'\nEvaluation: {len(train_features[0])}')

    dict_results = {}

    for reg_name, regressor in regressors_dict.items():

        print(f'\n  {reg_name}')

        regressor.fit(train_features, train_scores)

        dict_eval = evaluate(corpus_data, regressor, scaler, eval_measure)

        all_real_scores = dict_eval['real_scores']
        all_pred_scores = dict_eval['pred_scores']
        mean_real_score = dict_eval['mean_real_score']
        selected_descriptions = dict_eval['selected_descriptions']

        rmse = mean_squared_error(all_real_scores, all_pred_scores, squared=False)
        mae = mean_absolute_error(all_real_scores, all_pred_scores)
        r2 = r2_score(all_real_scores, all_pred_scores)
        pearson_corr = pearsonr(all_real_scores, all_pred_scores)
        kendall_tau = kendalltau(all_real_scores, all_pred_scores)

        print(f'\n    Mean real score: {mean_real_score}')

        print('    Mean RMSE:', rmse)
        print('    Mean MAE:', mae)
        print('    Mean R2 Score:', r2)
        print('    Mean Pearson Correlation:', pearson_corr)
        print('    Mean Kendall Tau:', kendall_tau)

        dict_results[reg_name] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'pearson_corr': pearson_corr,
            'kendall_tau': kendall_tau
        }

        file_desc_name = f'{reg_name}_{eval_measure}.txt'

        file_desc_name = file_desc_name.replace(' ', '_').lower()

        generated_desc_file = os.path.join(gen_desc_dir, file_desc_name)

        with open(generated_desc_file, 'w') as file:
            file.write('\n'.join(selected_descriptions))

    results_file_path = os.path.join(results_dir, f'ml_{corpus_name}_{eval_measure}.json')

    with open(results_file_path, 'w') as fp:
        json.dump(dict_results, fp, indent=4)
