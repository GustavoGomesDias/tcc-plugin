from enum import Enum
import javalang
import io
import tokenize
from token import tok_name
from src.utils.types.Language import Language

class Tokenize:
    def __init__(self, language: Language) -> None:
        self.language = language
        self.dict_type = dict()

    def count_token_python(self, code):
        buffer = io.StringIO(code)
        tokens = tokenize.generate_tokens(buffer.readline)
        for token in tokens:
            token_type = tok_name[token.type]

            if token_type not in self.dict_type.keys():
                self.dict_type[token_type] = 1
            else:
                self.dict_type[token_type] += 1

    def count_token_java(self, code):
        tokens = list(javalang.tokenizer.tokenize(code, ignore_errors=True))

        for token in tokens:
            token_type = str(token).split(' ')[0]
            if token_type not in self.dict_type.keys():
                self.dict_type[token_type] = 1
            else:
                self.dict_type[token_type] += 1

    def count_token_type(self, lst_code) -> dict:
        for code in lst_code:

            if self.language == Language.JAVA:
                self.count_token_java(code)
            else:
                self.count_token_python(code)
        return self.dict_type