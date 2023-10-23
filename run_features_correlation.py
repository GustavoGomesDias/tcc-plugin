import numpy as np

from src.meta_model.meta_utils import read_data, build_regression_data
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.target import FeatureCorrelation


if __name__ == '__main__':

    lang = 'java'
    # lang = 'python'

    corpus_name = 'huetal'
    # corpus_name = 'codexglue'
    # corpus_name = 'wanetal'

    eval_measure = 'rougel_f'
    # eval_measure = 'bleu_4_o'
    # eval_measure = 'meteor_score'

    corpus_path = f'features_files/{lang}/{corpus_name}/{corpus_name}_features.json'

    corpus_data = read_data(corpus_path)

    train_features, train_scores = build_regression_data(corpus_data, eval_measure)

    print(f'\nTrain Corpus: {len(corpus_data)} -- {len(train_features)}')

    scaler = MinMaxScaler()

    train_features = scaler.fit_transform(train_features)

    train_scores = np.asarray(train_scores)

    featues_names = list(corpus_data[0]['systems'][0]['features'].keys())

    visualizer = FeatureCorrelation(labels=featues_names, sort=False)

    visualizer.fit(train_features, train_scores)

    features_correlations = []

    for f, c in zip(visualizer.labels, visualizer.scores_):
        features_correlations.append((f, c))

    features_correlations.sort(key=lambda e: e[1], reverse=True)

    print('\nFeatures Correlation:\n')

    for f in features_correlations:
        print('  ', f)
