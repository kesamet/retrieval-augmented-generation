import json
from collections import deque

import tiktoken

from src import CFG


def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def trim_memory(latest: tuple[str, str], memory: deque[tuple[str, str]], num_tokens: deque[int]):
    memory.append(latest)
    num_tokens.append(num_tokens_from_string(json.dumps(latest)))
    while sum(num_tokens) > CFG.MAX_MEMORY_TOKENS:
        memory.popleft()
        num_tokens.popleft()
