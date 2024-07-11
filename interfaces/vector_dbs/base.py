from abc import ABC, abstractmethod
from typing import List, Any, Optional

class VectorDB(ABC):
    @abstractmethod
    def create_collection(self, collection_name: str, dimension: int):
        """
        Create a new collection in the vector database.

        Args:
            collection_name (str): The name of the collection to create.
            dimension (int): The dimension of the vectors to be stored in this collection.
        """
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str):
        """
        Delete a collection from the vector database.

        Args:
            collection_name (str): The name of the collection to delete.
        """
        pass

    @abstractmethod
    def insert(self, collection_name: str, id: str, vector: List[float], metadata: Optional[dict] = None):
        """
        Insert a vector into a collection.

        Args:
            collection_name (str): The name of the collection to insert into.
            id (str): A unique identifier for the vector.
            vector (List[float]): The vector to insert.
            metadata (Optional[dict]): Additional metadata to store with the vector.
        """
        pass

    @abstractmethod
    def search(self, collection_name: str, query_vector: List[float], k: int = 5) -> List[dict]:
        """
        Search for the k nearest neighbors of a query vector in a collection.

        Args:
            collection_name (str): The name of the collection to search in.
            query_vector (List[float]): The query vector.
            k (int): The number of nearest neighbors to return.

        Returns:
            List[dict]: A list of dictionaries, each containing the id, vector, metadata, and distance of a nearest neighbor.
        """
        pass

    @abstractmethod
    def delete_vector(self, collection_name: str, id: str):
        """
        Delete a vector from a collection.

        Args:
            collection_name (str): The name of the collection to delete from.
            id (str): The unique identifier of the vector to delete.
        """
        pass

    @abstractmethod
    def get_vector(self, collection_name: str, id: str) -> Optional[dict]:
        """
        Retrieve a vector and its metadata from a collection.

        Args:
            collection_name (str): The name of the collection to retrieve from.
            id (str): The unique identifier of the vector to retrieve.

        Returns:
            Optional[dict]: A dictionary containing the vector and metadata, or None if not found.
        """
        pass

    @abstractmethod
    def list_collections(self) -> List[str]:
        """
        List all collections in the vector database.

        Returns:
            List[str]: A list of collection names.
        """
        pass

    @abstractmethod
    def collection_info(self, collection_name: str) -> dict:
        """
        Get information about a specific collection.

        Args:
            collection_name (str): The name of the collection.

        Returns:
            dict: A dictionary containing information about the collection (e.g., size, dimension).
        """
        pass