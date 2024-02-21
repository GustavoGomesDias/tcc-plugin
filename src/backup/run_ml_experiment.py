import os
import numpy as np

from src.services.description.get_better_result import get_all_better_results
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

from src.utils.helpers import return_full_path


if __name__ == '__main__':

    json_files_dir = return_full_path('../../new_experiment/descriptions_json/')
    sorted_measure = 'rougel_f'

    datasets = [
        # ('java', 'codexglue'),
        ('java', 'huetal'),
        # ('python', 'codexglue'),
        ('python', 'wanetal'),
    ]

    print('\nRunning Experiment')

    classifiers = {
        'logistic_regression': LogisticRegression(max_iter=1000),
        'knn': KNeighborsClassifier(),
        'decision_tree': DecisionTreeClassifier(),
        'random_forest': RandomForestClassifier(),
        'extra_trees_classifier': ExtraTreesClassifier(),
        'xgboost': XGBClassifier(),
        'lgbm': LGBMClassifier(),
        'svc': SVC(),
        'mlp_classifier': MLPClassifier(max_iter=1000),
        'cat_boost_classifier': CatBoostClassifier(verbose=False)
    }

    for language, dataset_name in datasets:

        print(f'\n\tDataset: {language} -- {dataset_name}')

        json_path = os.path.join(json_files_dir, language, dataset_name, f'{dataset_name}.json')

        dataset = get_all_better_results(json_path=json_path, sorted_measure=sorted_measure)

        data_features = [list(example[0].values()) for example in dataset]
        labels = [example[1] for example in dataset]

        print('\n\t\tExample:')
        print(f'\t\t\tFeatures: {data_features[0]}')
        print(f'\t\t\tLabel: {labels[0]}')

        counter_labels = Counter(labels)

        print(f'\n\t\tLabels Distribution: {counter_labels}')

        label_encoder = LabelEncoder()
        scaler = MinMaxScaler()

        y = label_encoder.fit_transform(labels)

        data_features = scaler.fit_transform(data_features)

        X_train, X_test, y_train, y_test = train_test_split(
            data_features, y, stratify=y, test_size=0.1, random_state=42)

        print(f'\n\t\tTrain: {len(X_train)} -- {len(y_train)}')
        print(f'\t\tTest: {len(X_test)} -- {len(y_test)}')

        unique_labels = np.unique(y_train)

        class_weight = compute_class_weight(class_weight='balanced', classes=unique_labels,
                                            y=y_train)

        class_weight = dict(zip(unique_labels, class_weight))

        for clf_name, clf_base in classifiers.items():

            print(f'\n\t\tClassifier: {clf_name}')

            classifier = clone(clf_base)

            if hasattr(classifier, 'class_weight'):
                classifier.class_weight = class_weight

            classifier = Pipeline([
                ('feature_selection_1', VarianceThreshold(threshold=0.0)),
                ('classification', classifier)
            ])

            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(X_test)

            clf_report = classification_report(y_pred, y_test)

            print(f'\n\t\t\tEvaluation Report\n: {clf_report}')
