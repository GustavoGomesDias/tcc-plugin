from src.evaluation_measures.evaluation_measures import compute_rouge, compute_bleu, compute_meteor
import nltk

"""
    https://medium.com/explorations-in-language-and-learning/metrics-for-nlg-evaluation-c89b6a781054
"""

if __name__ == '__main__':

    nltk.download('punkt')
    nltk.download('wordnet')

    max_desc_len = 50

    reference = 'gets the value of the helpful votes property'
    candidate = 'gets the value of the reason type property'

    rouge_scores = compute_rouge(reference, candidate, max_desc_len)

    bleu_score = compute_bleu(reference, candidate)

    tokens_reference = reference.split()
    tokens_candidate = candidate.split()

    meteor_score = compute_meteor(tokens_reference, tokens_candidate)

    print(f'\nRouge: {rouge_scores}')

    print(f'\nBleu: {bleu_score}')

    print(f'\nMeteor: {meteor_score}')

