from abc import ABC, abstractmethod
from typing import List

from langchain.schema import Document


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, passages: List[Document]) -> List[Document]:
        """
        Reranks a list of passages based on a specific ranking algorithm.

        :param passages: A list of Passage objects representing the passages to be reranked.
        :type passages: List[Passage]
        :param query: str: The query that was used for retrieving the passages.
        :return: The reranked list of passages.
        :rtype: List[Passage]
        """
        pass
