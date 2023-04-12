import torch
import tokenize
import io
import javalang


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def python_tokenizer(line):
    result = []
    line = io.StringIO(line)
    for toktype, tok, start, end, line in tokenize.generate_tokens(line.readline):
        if not toktype == tokenize.COMMENT:
            if toktype == tokenize.STRING:
                result.append('CODE_STRING')
            elif toktype == tokenize.NUMBER:
                result.append('CODE_INTEGER')
            elif (not tok == '\n') and (not tok == '    '):
                result.append(str(tok))
    return ' '.join(result)


def tokenize_java_code(code):
    token_list = []
    tokens = list(javalang.tokenizer.tokenize(code))
    for token in tokens:
        token_list.append(token.value)
    return ' '.join(token_list)


def example_python(device_):

    model_name = 'SEBIS/code_trans_t5_base_code_documentation_generation_python_multitask_finetune'

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model = model.to(device_)

    tokenizer = AutoTokenizer.from_pretrained(model_name, skip_special_tokens=True)

    # code = 'def add_tensors(t, t1) -> Any:\n    return t + t1'
    # code = 'def sum(x, y):\n    return x + y'
    # code = 'def f(numbers, n):\n  if n not in numbers:\n  numbers.append(n)'
    code = """class DBConnectionError(Exception):

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.error_name = 'ConnectionError'"""

    tokenized_code = python_tokenizer(code)

    print('\nCode after tokenization:', tokenized_code)

    code_seq = tokenizer.encode(tokenized_code, return_tensors='pt', truncation=True, max_length=256)\

    code_seq = code_seq.to(device_)

    desc_ids = model.generate(code_seq, min_length=10, max_length=30, num_beams=10, early_stopping=True)

    description = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                   for g in desc_ids]

    description = description[0].strip()

    print('\nDescription:', description)


def example_java(device_):

    model_name = 'SEBIS/code_trans_t5_base_code_documentation_generation_java_multitask_finetune'

    tokenizer = AutoTokenizer.from_pretrained(model_name, skip_special_tokens=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model = model.to(device_)

    # code = "protected String renderUri(URI uri){\n  return uri.toASCIIString();\n}\n"
    code = 'public int mult(int x, int y) {\n  return x * y;\n}'

    print('\nCode:', code)

    tokenized_code = tokenize_java_code(code)

    print('\nTokenized code:', tokenized_code)

    code_seq = tokenizer.encode(tokenized_code, return_tensors='pt', truncation=True, max_length=256)

    code_seq = code_seq.to(device_)

    desc_ids = model.generate(code_seq, min_length=10, max_length=30, num_beams=10, early_stopping=True)

    description = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                   for g in desc_ids]

    description = description[0].strip()

    print('\nDescription:', description)


"""
    https://github.com/agemagician/CodeTrans
"""

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\nUsing device: {device}')

    print('\n\nPython Example:')

    example_python(device)

    print('\n\nJava Example:')

    example_java(device)
