from abc import ABC, abstractmethod
from typing import List, Any, Optional

class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, docs: List[dict], top_k: int = 5) -> List[dict]:
        """
        Rerank and return the top k results from your pipeline

        Args:
            docs (dict): List of documents with keys "id", "text", and optional key "metadata" to do document specific metadata
            query (str): The query related to these docs
            top_k (int): number of results to return

        """
        pass

    