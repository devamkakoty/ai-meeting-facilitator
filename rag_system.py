"""
RAG System Module for Sales Copilot using Pinecone and LangChain Google AI Embeddings.

This module implements a Retrieval Augmented Generation system
that searches a knowledge base for relevant information.
"""
import os
import time
import json
import logging
from typing import Dict, List, Any, Optional
import threading
from queue import Queue
import re
import uuid
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import Pinecone as LangchainPinecone, PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from utils.GenerateContentService import generate_content

logger = logging.getLogger("rag_system")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "googleServiceAccountCredentials.json"
os.environ["GOOGLE_API_KEY"] = "AIzaSyAnJg3id4YD6DT5wFAittSH_BGHv6mALvU"
os.environ["PINECONE_API_KEY"] = "pcsk_3DxeCk_EBLSPYzsFox6yaa6ngWSQ7QLZGkna9nt45WBzsVfZq6wKdhm3uPRNBA4xpjthLT"


class RAGSystem:
    """Implements a Retrieval Augmented Generation system with Pinecone and Google AI Embeddings."""

    def __init__(self,
                 knowledge_base_path: str = "KnowledgeBase",
                 pinecone_index_name: str = "copilot-knowledgebase",
                 pinecone_namespace: str = "default",
                 result_cache_size: int = 100,
                 use_metadata: bool = False):
        """Initialize the RAG system.

        Args:
            knowledge_base_path: Path to the knowledge base files.
            pinecone_index_name: Name of the Pinecone index to use.
            pinecone_namespace: Namespace within the Pinecone index.
            result_cache_size: Maximum number of results to cache.
            use_metadata: Whether to use metadata for improved retrieval.
        """
        self.knowledge_base_path = knowledge_base_path
        self.pinecone_index_name = pinecone_index_name
        self.vector_store = None
        self.result_cache_size = result_cache_size
        self.use_metadata = use_metadata
        self.pinecone_index = None
        self.embeddings_model = None

        # Initialize local document tracking
        self.documents = []
        self.metadata = []
        self.result_cache = {}
        self.id_to_doc_map = {}  # Maps Pinecone IDs to local document indexes

        # Queue for asynchronous processing
        self.query_queue = Queue()

        # Initialize Pinecone and load documents
        self.initialize_vector_store()

        # Start background worker thread
        self.worker_thread = threading.Thread(target=self._process_query_queue, daemon=True)
        self.worker_thread.start()

        logger.info(f"RAG System initialized with knowledge base at: {knowledge_base_path}")

    def initialize_vector_store(self):
        """Initialize Pinecone and load documents from the knowledge base."""
        try:
            # Check if knowledge base directory exists
            if not os.path.exists(self.knowledge_base_path):
                os.makedirs(self.knowledge_base_path)
                logger.info(f"Created knowledge base directory: {self.knowledge_base_path}")

            try:
                # Initialize the LangChain embeddings model
                self.embeddings_model = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=os.environ["GOOGLE_API_KEY"]
                )

                logger.info("Initialized LangChain Google AI embedding client")

                # Initialize Pinecone
                pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
                if self.pinecone_index_name not in pc.list_indexes().names():
                    sample_embedding = self._get_embedding("This is a sample text for dimension detection")
                    dimension = len(sample_embedding)
                    pc.create_index(self.pinecone_index_name, dimension=dimension, metric="cosine", spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    ))

                    # wait for index to be initialized
                    while not pc.describe_index(self.pinecone_index_name).status['ready']:
                        time.sleep(1)

                    logger.info(f"Created new Pinecone index: {self.pinecone_index_name} with dimension {dimension}")

                # connect to index
                self.pinecone_index = pc.Index(self.pinecone_index_name)

                # Prepare vectors for upsert
                vectors = []
                (logger.info
                 (f"Connected to Pinecone index: {self.pinecone_index_name}"))

                # Load documents from knowledge base
                for filename in os.listdir(self.knowledge_base_path):
                    if filename.endswith('.json'):
                        try:
                            with open(os.path.join(self.knowledge_base_path, filename), 'r') as f:
                                data = json.load(f)

                                # Process each document in the file
                                for doc in data.get('documents', []):
                                    category = doc.get("category", '')
                                    title = doc.get("title", '')
                                    text =  doc.get('content', '')
                                    tags = doc.get('metadata', [])
                                    doc_id = doc.get('id', f"doc_{uuid.uuid4()}")

                                    # Skip empty documents
                                    if not doc.get('content', '').strip():
                                        continue

                                    # Combine relevant text for embedding
                                    text_to_embed = f"{doc['title']}: {doc['content']}"

                                    # Get embedding
                                    embedding = self._get_embedding(text_to_embed)

                                    # Prepare metadata
                                    metadata = {
                                        "category": category,
                                        "title": title,
                                        "text": text,
                                        "tags": tags,
                                    }

                                    # Create vector record
                                    vector = {
                                        "id": doc["id"],
                                        "values": embedding,
                                        "metadata": metadata
                                    }
                                    # Upsert the vectors into Pinecone using LangChain
                                    vectors.append(vector)

                                # Upload vectors to Pinecone in batches
                                batch_size = 100  # Adjust based on needs
                                for i in range(0, len(vectors), batch_size):
                                    batch = vectors[i:i + batch_size]
                                    self.pinecone_index.upsert(vectors=batch)

                                print(f"Successfully uploaded {len(vectors)} vectors to Pinecone index '{self.pinecone_index_name}'")
                        except Exception as e:
                            logger.error(f"Error loading document {filename}: {e}")

                logger.info(f"Loaded {len(self.documents)} documents into RAG system")

            except ImportError as e:
                logger.warning(f"Required packages not installed: {e}")
                logger.warning("RAG system will use text-based search as fallback")

        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using LangChain Google AI Embeddings.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        try:
            # Use LangChain's embed_query method
            embedding = self.embeddings_model.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding from LangChain Google AI: {e}")
            # Return zero vector of appropriate size as fallback
            return [0.0] * 768  # Default dimension for embedding models

    def add_query(self,
                  query_text: str,
                  topics: List[str] = None,
                  callback: callable = None) -> None:
        """Queue a query for asynchronous processing.

        Args:
            query_text: The query text.
            topics: List of topics to focus the search.
            callback: Function to call with results.
        """
        # Check cache first
        cache_key = f"{query_text}:{','.join(topics) if topics else ''}"
        if cache_key in self.result_cache:
            result = self.result_cache[cache_key]
            if callback:
                callback(result)
            return result

        # Add to processing queue
        self.query_queue.put({
            'query': query_text,
            'topics': topics,
            'callback': callback,
            'timestamp': time.time()
        })

    def _process_query_queue(self):
        """Process queries from the queue in a background thread."""
        while True:
            item = self.query_queue.get()
            try:
                # Process the query
                result = self.search(
                    query_text=item['query'],
                    topics=item['topics']
                )

                # Cache the result
                cache_key = f"{item['query']}:{','.join(item['topics']) if item['topics'] else ''}"
                self.result_cache[cache_key] = result

                # Trim cache if too large
                if len(self.result_cache) > self.result_cache_size:
                    # Remove oldest entries
                    oldest_keys = sorted(self.result_cache.keys())[:len(self.result_cache) - self.result_cache_size]
                    for key in oldest_keys:
                        del self.result_cache[key]

                # Call the callback with results
                if item['callback']:
                    item['callback'](result)

            except Exception as e:
                logger.error(f"Error processing RAG query: {e}")
                # Send empty result on error
                if item['callback']:
                    item['callback']({
                        'query': item['query'],
                        'results': [],
                        'message': f"Error processing query: {str(e)}"
                    })

            finally:
                self.query_queue.task_done()

    def search(self, query_text: str, topics: List[str] = None) -> Dict[str, Any]:
        """Search the knowledge base for relevant information.

        Args:
            query_text: The query text.
            topics: List of topics to focus the search.

        Returns:
            Dictionary with search results.
        """
        start_time = time.time()

        try:
            if self.pinecone_index is not None:
                try:
                    # Get embedding for the query
                    query_embedding = self._get_embedding(query_text)

                    # Search options
                    top_k = 5

                    # If topics are provided and we want to use metadata, add a filter
                    filter_condition = None
                    if topics and self.use_metadata:
                        # Create filter for topic matches
                        # Assuming tags contains topics
                        topic_filter = {"tags": {"$in": topics}}
                        filter_condition = topic_filter

                    # Search using Pinecone directly
                    search_results = self.pinecone_index.query(
                        vector=query_embedding,
                        top_k=top_k,
                        filter=filter_condition,
                        include_metadata=True
                    )

                    # Format results
                    processed_results = []
                    for match in search_results.matches:
                        if match.score > 0.3:  # Assuming score is similarity, adjust threshold as needed
                            processed_result = {
                                'id': match.id,
                                'score': match.score,
                                'text': match.metadata.get('text', ''),
                                'metadata': {
                                    'category': match.metadata.get('category', ''),
                                    'title': match.metadata.get('title', ''),
                                    'tags': match.metadata.get('tags', [])
                                }
                            }
                            processed_results.append(processed_result)

                    return {
                        'query': query_text,
                        'results': processed_results,
                        'message': "Success (Pinecone search)",
                        'elapsed_time': time.time() - start_time
                    }

                except Exception as e:
                    logger.error(f"Pinecone search failed, falling back to text search: {e}")

            # Return message if unsuccessful
            return {
                'query': query_text,
                'results': [],
                'message': "Failure (no vector search available)",
                'elapsed_time': time.time() - start_time
            }

        except Exception as e:
            logger.error(f"Search error: {e}")
            return {
                'query': query_text,
                'results': [],
                'message': f"Error: {str(e)}",
                'elapsed_time': time.time() - start_time
            }

    def generate_response(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Generate a response using Gemini based on search results.

        Args:
            query: Original user query.
            search_results: List of search results from Pinecone.

        Returns:
            Generated response from Gemini.
        """
        try:
            # Prepare context from search results
            context = "\n\n".join([
                f"[{result.get('title', 'Untitled')}]: {result.get('content', '')}"
                for result in search_results
            ])

            # Construct prompt for Gemini
            prompt = f"""
            Context Information:
            {context}

            User Query: {query}

            Based on the provided context, generate a comprehensive and helpful response to the user's query. 
            If the context doesn't fully answer the query, provide the best possible answer using the available information. 
            Be clear, concise, and directly address the user's question.
            """

            # Generate response using Gemini
            response = generate_content(prompt)

            return response.text

        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return f"I apologize, but I encountered an error generating a response: {str(e)}"

    def process_query(self, query: str, topics: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive method to search and generate a response.

        Args:
            query: User's query.
            topics: Optional list of topics to filter search.

        Returns:
            Dictionary with search results and generated response.
        """
        # Perform semantic search
        search_result = self.search(query, topics)

        # Generate response based on search results
        response = self.generate_response(query, search_result.get('results', []))

        # Combine search result with response
        search_result['response'] = response

        return search_result

ob = RAGSystem()
print(ob.search("Should I open a Checking Account?"))
