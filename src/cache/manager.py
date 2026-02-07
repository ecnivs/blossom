import logging
import chromadb
from chromadb.utils import embedding_functions
from config import config
import os


class SemanticCacheManager:
    """
    Manages semantic caching of LLM responses using ChromaDB and Google GenAI embeddings.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.enabled = config.cache.enabled

        if not self.enabled:
            self.logger.info("Semantic caching is disabled.")
            return

        try:
            self.chroma_client = chromadb.PersistentClient(
                path=config.cache.persist_directory
            )

            # Use local sentence-transformers embeddings
            self.embedding_fn = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            )

            self.collection = self.chroma_client.get_or_create_collection(
                name="llm_responses",
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},  # Force cosine distance
            )
            self.logger.info(
                f"Cache initialized with local embeddings at {config.cache.persist_directory}"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize cache: {e}")
            self.enabled = False

    def get(self, query: str) -> str | None:
        """
        Retrieve a cached response if a semantic match is found.
        """
        if not self.enabled:
            return None

        try:
            results = self.collection.query(query_texts=[query], n_results=1)

            if not results["documents"] or not results["documents"][0]:
                return None

            distance = results["distances"][0][0]

            self.logger.info(
                f"Cache check: '{query}' -> Top match distance: {distance:.4f}"
            )

            # Cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite
            # Use strict threshold to avoid matching different questions
            dist_threshold = 0.15

            if distance < dist_threshold:
                cached_response = results["documents"][0][0]
                self.logger.info(
                    f"Cache hit! Distance {distance:.4f} < {dist_threshold}"
                )
                return cached_response
            else:
                self.logger.info(
                    f"Cache miss. Distance {distance:.4f} >= {dist_threshold}"
                )

            return None

        except Exception as e:
            self.logger.error(f"Error querying cache: {e}")
            return None

    def add(self, query: str, response: str):
        """
        Add a query-response pair to the cache.
        """
        if not self.enabled:
            return

        if not self.should_cache(response):
            return

        try:
            # Generate a unique ID (hash of query or just uuid)
            import hashlib

            doc_id = hashlib.md5(query.encode()).hexdigest()

            self.collection.upsert(
                documents=[response],
                ids=[doc_id],
                metadatas=[{"query": query}],  # store original query in metadata
            )
            self.logger.debug(f"Cached response for query: {query[:50]}...")

        except Exception as e:
            self.logger.error(f"Error adding to cache: {e}")

    def should_cache(self, response_text: str) -> bool:
        """
        Determine if a response is safe to cache.
        """
        # Do not cache responses that invoke tools (plugins)
        if "PLUGIN:" in response_text:
            return False

        # Do not cache error messages potentially
        if "timeout" in response_text.lower() or "error" in response_text.lower():
            pass

        return True
