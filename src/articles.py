import csv
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO, Iterator


#Data labels
LABELS = ['politics', 'sport', 'tech', 'business', 'entertainment']

#Rebuilding data path
DATA_PATH = Path(Path.cwd().parent, "data")

@contextmanager
def load_file(relative_path: str | Path, encoding: str = "utf-8") -> Iterator[TextIO]:
    """Load file"""
    f = (DATA_PATH / relative_path).expanduser().open("r", encoding=encoding, newline="")
    try:
        yield f
    finally:
        f.close()

def load_dataset(path: str, delimiter: str = ",") -> list:
    """Load dataset ArticleId,Text"""
    with load_file(path) as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def load_dataset_per_category(path: str,) -> dict:
    """Load training dataset per category for training purposes."""

    training_dataset = load_dataset(path)[0:1000] # Limit to 1000 for training

    articles_per_category = defaultdict(list) # category: [(article_id, text),]
    for article in training_dataset:
        articles_per_category[article['Category']].append((article['ArticleId'],article['Text']))

    return articles_per_category
