from src.evaluation_measures.evaluation_measures import compute_bert_score

code = """class DBConnectionError(Exception):

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.error_name = 'ConnectionError'"""


reference = "T"
candidate = "Exception class for DBConnectionError."

print(compute_bert_score(reference, candidate   ))