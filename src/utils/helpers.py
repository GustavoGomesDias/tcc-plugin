import os
from pathlib import Path

def return_full_path(path_complement):
    return os.path.join(Path(__file__).parent.resolve(), '..', '..', path_complement)