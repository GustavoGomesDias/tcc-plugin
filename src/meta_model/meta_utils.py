import json
import re
import sys

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
from tqdm import tqdm


def read_data(file_path):
    with open(file_path) as data_file:
        all_data = json.load(data_file)
    return all_data


def compute_jaccard(tokens_1, tokens_2):
    set_1 = set(tokens_1)
    set_2 = set(tokens_2)
    set_inter = set_1.intersection(set_2)
    set_union = set_1.union(set_2)
    if len(set_inter) == 0 or len(set_union) == 0:
        return 0
    similarity = len(set_inter) / len(set_union)
    return similarity


def compute_rouge(reference, candidate, max_len, use_stemming=True, max_ngram=2):
    evaluator = Rouge(metrics=['rouge-n', 'rouge-l'], max_n=max_ngram, limit_length=True,
                      length_limit=max_len, length_limit_type='words', apply_avg=True,
                      apply_best=False, alpha=0.5, weight_factor=1.0, stemming=use_stemming)
    rouge_scores = evaluator.get_scores(candidate, reference)
    return rouge_scores


def process_data(train_data):
    all_descs = []
    for data in train_data:
        for system_data in data['systems']:
            desc = system_data['description']
            desc = re.sub(r'[^\w\s]', '', desc)
            all_descs.append(desc)
    tfidf_vectorizer = TfidfVectorizer()
    binary_vectorizer = CountVectorizer(binary=True)
    count_vectorizer = CountVectorizer(binary=False)
    tfidf_vectorizer.fit(all_descs)
    binary_vectorizer.fit(all_descs)
    count_vectorizer.fit(all_descs)
    return tfidf_vectorizer, binary_vectorizer, count_vectorizer


def extract_features(test_data, train_data, max_desc_len):

    tfidf_vectorizer, binary_vectorizer, count_vectorizer = process_data(train_data)

    with tqdm(total=len(test_data), file=sys.stdout, colour='red', desc='Extracting features') as pbar:

        for data in test_data:

            code = data['code']

            tokens_code = [t for t in code.split() if len(t) > 1]

            len_code = len(tokens_code)

            for system_data in data['systems']:

                desc = system_data['description']

                desc = re.sub(r'[^\w\s]', '', desc)

                words_desc = [t for t in desc.split() if len(t) > 1]

                len_desc = len(words_desc)

                if len_desc > 0:

                    ratio_len = len_desc / len_code

                    desc_tfid = tfidf_vectorizer.transform([desc])
                    desc_binary = binary_vectorizer.transform([desc])
                    desc_count = count_vectorizer.transform([desc])

                    mean_jacc_sim = 0

                    mean_cosine_sim_tfidf = 0
                    mean_cosine_sim_binary = 0
                    mean_cosine_sim_count = 0

                    rouge_scores = {}

                    for other_system_data in data['systems']:

                        if system_data['name'] != other_system_data['name']:

                            other_desc = other_system_data['description']

                            other_desc = re.sub(r'[^\w\s]', '', other_desc)

                            other_words_desc = [t for t in other_desc.split() if len(t) > 1]

                            mean_jacc_sim += compute_jaccard(words_desc, other_words_desc)

                            other_desc_tfidf = tfidf_vectorizer.transform([other_desc])
                            other_desc_binary = binary_vectorizer.transform([other_desc])
                            other_desc_count = count_vectorizer.transform([other_desc])

                            cosine_sim_tfidf = cosine_similarity(desc_tfid, other_desc_tfidf)[0][0]
                            cosine_sim_binary = cosine_similarity(desc_binary, other_desc_binary)[0][0]
                            cosine_sim_count = cosine_similarity(desc_count, other_desc_count)[0][0]

                            mean_cosine_sim_tfidf += cosine_sim_tfidf
                            mean_cosine_sim_binary += cosine_sim_binary
                            mean_cosine_sim_count += cosine_sim_count

                            r_scores = compute_rouge(desc, other_desc, max_desc_len)

                            for rouge_n, metrics in r_scores.items():
                                for metric, value in metrics.items():
                                    metric_name = rouge_n.replace('-', '') + '_' + metric
                                    if metric_name in rouge_scores:
                                        rouge_scores[metric_name].append(value)
                                    else:
                                        rouge_scores[metric_name] = [value]

                    if len(data['systems']) >= 2:

                        mean_jacc_sim /= (len(data['systems']) - 1)
                        mean_cosine_sim_tfidf /= (len(data['systems']) - 1)
                        mean_cosine_sim_binary /= (len(data['systems']) - 1)
                        mean_cosine_sim_count /= (len(data['systems']) - 1)

                    features = {
                        'len_code': len_code,
                        'len_desc': len_desc,
                        'ratio_len': ratio_len,
                        'mean_jacc_sim': mean_jacc_sim,
                        'mean_cosine_sim_tfidf': mean_cosine_sim_tfidf,
                        'mean_cosine_sim_binary': mean_cosine_sim_binary,
                        'mean_cosine_sim_count': mean_cosine_sim_count
                    }

                    for rouge_n, values in rouge_scores.items():
                        features[rouge_n] = sum(values) / len(values)

                    system_data['features'] = features

                else:

                    features = {
                        'len_code': 0,
                        'len_desc': 0,
                        'ratio_len': 0,
                        'tokens_freq': 0,
                        'words_freq': 0,
                        'mean_previous_eval': 0,
                        'mean_jacc_sim': 0,
                        'mean_cosine_sim': 0
                    }

                    system_data['features'] = features

            pbar.update(1)


def evaluate(corpus_data, regressor, scaler, eval_measure):
    all_real_scores = []
    all_pred_scores = []
    mean_real_score = 0
    selected_descriptions = []
    for example in corpus_data:
        predictions = []
        for system_data in example['systems']:
            if system_data['features']['len_desc'] > 0:
                features = []
                for feature_name, feature_value in system_data['features'].items():
                    if feature_name not in ['mean_previous_eval', 'tokens_freq', 'words_freq']:
                        features.append(feature_value)
                features = scaler.transform([features])
                y_pred = regressor.predict(features)[0]
                y_true = system_data['measures'][eval_measure]
                predictions.append((y_pred, y_true, system_data['description']))
                all_real_scores.append(y_true)
                all_pred_scores.append(y_pred)
        predictions.sort(key=lambda x: x[0], reverse=True)
        selected_system = predictions[0]
        mean_real_score += selected_system[1]
        selected_descriptions.append(selected_system[2])
    mean_real_score /= len(corpus_data)
    dict_eval = {
        'real_scores': all_real_scores,
        'pred_scores': all_pred_scores,
        'mean_real_score': mean_real_score,
        'selected_descriptions': selected_descriptions
    }
    return dict_eval


def build_regression_data(corpus_data, eval_measure_name):
    features_corpus = []
    eval_scores_corpus = []
    for data in corpus_data:
        for system_data in data['systems']:
            if system_data['features']['len_desc'] == 0:
                continue
            eval_measure_score = system_data['measures'][eval_measure_name]
            features = []
            for feature_name, feature_value in system_data['features'].items():
                if feature_name not in ['mean_previous_eval', 'tokens_freq', 'words_freq']:
                    features.append(feature_value)
            features_corpus.append(features)
            eval_scores_corpus.append(eval_measure_score)
    return features_corpus, eval_scores_corpus
