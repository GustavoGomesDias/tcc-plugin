import torch

from transformers import PLBartTokenizer, PLBartForConditionalGeneration

"""
    https://github.com/wasiahmad/PLBART
"""

if __name__ == '__main__':

    # lang = 'java'
    lang = 'python'

    if lang == 'java':
        model_name = 'uclanlp/plbart-java-en_XX'
    else:
        model_name = 'uclanlp/plbart-python-en_XX'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\nUsing device: {device}')

    tokenizer = PLBartTokenizer.from_pretrained(model_name, src_lang=lang, tgt_lang='__en_XX__')
    model = PLBartForConditionalGeneration.from_pretrained(model_name)

    model = model.to(device)

    if lang == 'java':
        code = 'public int mult ( int x , int y ) { return x * y ; }'
    else:
        code = 'def f(numbers, n):\n  if n not in numbers:\n  numbers.append(n)'

    print(f'\nCode: {lang} -- {code}')

    start_token_id = tokenizer.lang_code_to_id['__en_XX__']

    inputs = tokenizer(code, return_tensors='pt', max_length=200, truncation=True)

    inputs = inputs.to(device)

    translated_tokens = model.generate(**inputs, decoder_start_token_id=start_token_id, max_length=30,
                                       num_beams=10)

    desc = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    print(f'\nDescription: {desc}')