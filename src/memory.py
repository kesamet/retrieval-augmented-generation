import json
from collections import deque

from src import CFG


def num_words_from_string(string: str) -> int:
    """Count number of words in string."""
    num_words = len(string.split())
    return num_words


def trim_memory(latest: tuple[str, str], memory: deque[tuple[str, str]], num_words: deque[int]):
    memory.append(latest)
    num_words.append(num_words_from_string(json.dumps(latest)))
    while sum(num_words) > CFG.MAX_MEMORY_WORDS:
        memory.popleft()
        num_words.popleft()
