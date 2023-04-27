import javalang
import io
import tokenize


if __name__ == '__main__':

    codes_python = [
        'def add_tensors(t, t1) -> Any:\n    return t + t1',
        'def sum(x, y):\n    return x + y',
        'def f(numbers, n):\n  if n not in numbers:\n  numbers.append(n)'
    ]

    codes_java = [
        'public int mult(int x, int y) {\n  return x * y;\n}',
        'public int hashcode ( ) { return value . hashcode ( ) ; }'
        'public static double get Similarity(String phrase1, String phrase2) {\nreturn (get SC(phrase1, phrase2) + '
        'getSC(phrase2, phrase1)) / 2.0;'
    ]

    print('\n\n---------------- Python ----------------\n')

    for code in codes_python:

        code = code.replace('\n', ' ').strip()

        print(f'\nCode: {code}\n')

        buffer = io.StringIO(code)

        for type_, tok, _, _, _ in tokenize.generate_tokens(buffer.readline):
            print(f'  Tokens: {type_} - {tok}')

    print('\n\n---------------- Java ----------------\n')

    for code in codes_java:

        code = code.replace('\n', ' ').strip()

        print(f'\nCode: {code}\n')

        tokens = list(javalang.tokenizer.tokenize(code, ignore_errors=True))

        for token in tokens:
            print(f'  Tokens: {token}')
