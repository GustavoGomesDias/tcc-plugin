import torch

from transformers import RobertaTokenizer, T5ForConditionalGeneration


"""
    https://github.com/salesforce/CodeT5
"""

if __name__ == '__main__':

    model_name_tok = 'Salesforce/codet5-base'
    model_name = 'Salesforce/codet5-base-multi-sum'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\nUsing device: {device}')

    tokenizer = RobertaTokenizer.from_pretrained(model_name_tok)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    model = model.to(device)

    # code = "def greet(user): print(f'hello <extra_id_0>!')"
    # code = 'def add_tensors(t, t1) -> Any:    return t + t1'
    # code = 'def sum(x, y):\n    return x + y'
    # code = 'def f(numbers, n):\n  if n not in numbers:\n  numbers.append(n)'
    # code = "protected String renderUri(URI uri){\n  return uri.toASCIIString();\n}\n"
    # code = 'public int mult(int x, int y) {\n  return x * y;\n}'

    code = """class DBConnectionError(Exception):

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.error_name = 'ConnectionError'"""

    input_ids = tokenizer(code, return_tensors='pt').input_ids

    input_ids = input_ids.to(device)

    generated_ids = model.generate(input_ids, max_length=30)

    desc = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print('\nDescription:', desc)
