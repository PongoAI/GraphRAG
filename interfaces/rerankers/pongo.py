from .base import Reranker
import pongo
from typing import List

class PongoReranker(Reranker):
    def __init__(self, pongo_secret):
        self.client = pongo.PongoClient(pongo_secret)

    def rerank(self, query: str, docs: List[dict], top_k: int = 5) -> List[dict]:
        reranked_docs = self.client.filter(query, docs, num_results=top_k).json()
        return reranked_docs
