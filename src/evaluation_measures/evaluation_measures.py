from rouge import Rouge
from bert_score import BERTScorer
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


"""
    https://github.com/Diego999/py-rouge

    Bert Score
    https://github.com/Tiiiger/bert_score
    https://arxiv.org/pdf/1904.09675.pdf
    https://pypi.org/project/bert-score/
"""


def compute_rouge(reference, candidate, max_len, use_stemming=True, max_ngram=2):
    evaluator = Rouge(metrics=['rouge-n', 'rouge-l'], max_n=max_ngram, limit_length=True,
                      length_limit=max_len,
                      length_limit_type='words', apply_avg=True, apply_best=False, alpha=0.5,
                      weight_factor=1.0,
                      stemming=use_stemming)
    rouge_scores_ = evaluator.get_scores(candidate, reference)
    return rouge_scores_


def compute_bleu(reference_, candidate):
    smooth = SmoothingFunction()
    return sentence_bleu([reference_.split()], candidate.split(),
                         smoothing_function=smooth.method2)


def compute_meteor(tokens_reference, tokens_candidate):
    return single_meteor_score(tokens_reference, tokens_candidate)


def compute_bert_score(tokens_reference: str, tokens_candidate: str):
    score = BERTScorer(lang='en', batch_size=3)
    precision, recall, f_measure = score.score(cands=[tokens_candidate], refs=[tokens_reference])
    return precision.numpy()[0], recall.numpy()[0], f_measure.numpy()[0]
