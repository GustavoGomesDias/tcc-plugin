import torch
import src.utils.code_bert_utils as utils

from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

"""
    https://github.com/microsoft/CodeBERT
    
    Modelo pré-treinado:
        https://code-summary.s3.amazonaws.com/pytorch_model.bin
"""

if __name__ == '__main__':
    try:
        config = RobertaConfig.from_pretrained('microsoft/codebert-base')

        model_file = '/home/gustavo/dev/faculdade/tcc-plugin/models/codebert/pytorch_model.bin'

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f'\nUsing device: {device}')

        tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base', do_lower_case=False)

        model = utils.build_model(model_class=RobertaModel, model_file=model_file, config=config,
                                tokenizer=tokenizer, max_len=30, beam_size=10,)

        model = model.to(device)

        # code = 'def add_tensors(t, t1) -> Any:\n    return t + t1'
        # code = 'def sum(x, y):\n    return x + y'
        # code = 'public int mult(int x, int y) {\n  return x * y;\n}'
        # code = 'def f(numbers, n):\n  if n not in numbers:\n  numbers.append(n)'

        # code = 'public int hashcode ( ) { return value . hashcode ( ) ; }'

        # code = """public static double get Similarity(String phrase1, String phrase2) {
        #     return (get SC(phrase1, phrase2) + getSC(phrase2, phrase1)) / 2.0;
        # }"""

        code = """class DBConnectionError(Exception):

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.error_name = 'ConnectionError'"""

        example = [utils.Example(source=code, target=None)]

        features_code = utils.get_features(example, tokenizer, max_code_len=300)

        description, length = utils.inference(features_code, model, tokenizer, device)

        print('\nDescription:', description[0])
    except Exception as e:
        print(e)