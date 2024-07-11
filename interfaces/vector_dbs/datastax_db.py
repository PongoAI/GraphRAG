from typing import List, Optional
from astrapy import DataAPIClient
from astrapy.constants import VectorMetric
from astrapy.ids import UUID
from astrapy.info import CollectionVectorServiceOptions
import uuid
from .base import VectorDB
import os
from dotenv import load_dotenv

load_dotenv()

class DatastaxDB(VectorDB):
    def __init__(self):
        api_endpoint = os.getenv('DATASTAX_API_ENDPOINT')
        token = os.getenv('DATASTAX_TOKEN')
        if not api_endpoint or not token:
            raise ValueError("DATASTAX_API_ENDPOINT and DATASTAX_TOKEN must be set in .env file")
        self.client = DataAPIClient(token)
        self.database = self.client.get_database_by_api_endpoint(api_endpoint)
        self._collection_cache = {}

    def create_collection(self, collection_name: str, dimension: int):
        self.database.create_collection(
            collection_name,
            metric=VectorMetric.COSINE,
            service=CollectionVectorServiceOptions(
                provider="openai",
                model_name="text-embedding-3-small",
                authentication={
                    "providerKey": "openai_secret",
                },
            ),
        )
        self._collection_cache[collection_name] = self.database.get_collection(collection_name)

    def delete_collection(self, collection_name: str):
        self.database.delete_collection(collection_name)
        self._collection_cache.pop(collection_name, None)

    def _get_collection(self, collection_name: str):
        if collection_name not in self._collection_cache:
            self._collection_cache[collection_name] = self.database.get_collection(collection_name)
        return self._collection_cache[collection_name]

    def insert(self, collection_name: str, id: str, vector: List[float], metadata: Optional[dict] = None):
        collection = self._get_collection(collection_name)
        document = {
            '_id': UUID(id),
            'vector': vector,
            **metadata
        }
        collection.insert_one(document)

    def search(self, collection_name: str, query: str, k: int = 5) -> List[dict]:
        collection = self._get_collection(collection_name)
        results = collection.find(
            sort={"$vectorize": query},
            limit=k,
            projection={"$vectorize": True},
            include_similarity=True,
        )


        return [
            {
                'id': str(doc['_id']),
                'text': doc['$vectorize'],
                'score': doc.get('$similarity')
            }
            for doc in results
        ]

    def delete_vector(self, collection_name: str, id: str):
        collection = self._get_collection(collection_name)
        collection.delete_one({'_id': UUID(id)})

    def get_vector(self, collection_name: str, id: str) -> Optional[dict]:
        collection = self._get_collection(collection_name)
        doc = collection.find_one({'_id': UUID(id)})
        if doc:
            return {
                'id': str(doc['_id']),
                'vector': doc.get('vector'),
                'metadata': {k: v for k, v in doc.items() if k not in ['_id', 'vector']}
            }
        return None

    def list_collections(self) -> List[str]:
        return self.database.get_collections()

    def collection_info(self, collection_name: str) -> dict:
        collection = self._get_collection(collection_name)
        info = collection.get_info()
        return {
            'name': info.name,
            'size': info.count,
            'dimension': info.options.vector_dimension
        }
