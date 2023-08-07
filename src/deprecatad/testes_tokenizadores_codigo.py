import javalang
import io
import tokenize
from token import tok_name
from src.services.tokens.Tokenize import Tokenize, Language

if __name__ == '__main__':

    codes_python = [
        'def add_tensors(t, t1) -> Any:\n    return t + t1',
        # 'def sum(x, y):\n    return x + y',
        # 'def f(numbers, n):\n  if n not in numbers:\n  numbers.append(n)'
    ]

    codes_java = [
        'public int mult(int x, int y) {\n  return x * y;\n}',
        # 'public int hashcode ( ) { return value . hashcode ( ) ; }'
        # 'public static double get Similarity(String phrase1, String phrase2) {\nreturn (get SC(phrase1, phrase2) + '
        # 'getSC(phrase2, phrase1)) / 2.0;'
    ]

    """
        Características para extrair do código:
            
            1. Total de tokens;
            2. Dicionário com os tipos de tokens possíveis como chave e a frequência no código como valor;
                {
                    tipo_1: freq_1,
                    tipo_2: freq_2,
                    ...,
                    tipo_n: freq_n
                }            
    
        T1: Eu adorei o filme ontem ==> Positivo
        T2. Não gostei do filme     ==> Negativo
        
        Vocabulário: [eu, adorei, o, filme, ontem, nao, gostei, do]
        
        Bag of words:
        
            T1: [1,1,1,1,1,0,0,0]
            T2: [0,0,0,1,0,1,1,1]
    """

    print('\n\n---------------- Python ----------------\n')

    for code in codes_python:

        code = code.replace('\n', ' ').strip()

        print(f'\nCode: {code}\n')

        buffer = io.StringIO(code)

        tokens = tokenize.generate_tokens(buffer.readline)

        for token in tokens:
            token_type = tok_name[token.type]
            print(f'  Tokens: {token_type} - {token.string} - {token.start} - {token.end}')

    # print('\n\n---------------- Java ----------------\n')

    # for code in codes_java:

    #     code = code.replace('\n', ' ').strip()

    #     print(f'\nCode: {code}\n')

    #     tokens = list(javalang.tokenizer.tokenize(code, ignore_errors=True))

    #     for token in tokens:
    #         t = str(token).split(' ')[0]
    #         print(f'  Tokens: {t}')

    tokenize_service = Tokenize(Language.PYTHON)

    print(tokenize_service.count_token_type(codes_python))
