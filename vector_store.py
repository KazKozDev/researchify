from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Data class for storing document information."""
    id: str
    title: str
    authors: str
    abstract: str
    pdf_link: Optional[str]
    arxiv_link: Optional[str]
    published: str
    categories: str
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self):
        """Convert to dictionary, excluding the embedding."""
        d = asdict(self)
        d.pop('embedding', None)
        return d

class VectorStore:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: str = "vector_store",
        dimension: int = 384,
        batch_size: int = 32
    ):
        """
        Initialize the vector store with FAISS index and sentence transformer.
        
        Args:
            model_name: Name of the sentence transformer model
            index_path: Path to store the index and metadata
            dimension: Dimension of the embeddings
            batch_size: Batch size for processing documents
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = dimension
        self.batch_size = batch_size
        self.index_path = index_path
        self.metadata_path = os.path.join(index_path, "metadata.pkl")
        self.index_file = os.path.join(index_path, "faiss.index")
        
        # Create directory if it doesn't exist
        os.makedirs(index_path, exist_ok=True)
        
        # Initialize or load the index
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(dimension)
            self.metadata = {}
        
        # Thread lock for safe concurrent access
        self.lock = threading.Lock()
        
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.model.encode(text, show_progress_bar=False)
    
    def _batch_generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        return self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=True)
    
    def add_documents(self, documents: List[Document], update_existing: bool = True) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects
            update_existing: Whether to update existing documents with the same ID
        """
        with self.lock:
            # Filter out existing documents if not updating
            if not update_existing:
                documents = [doc for doc in documents if doc.id not in self.metadata]
            
            if not documents:
                return
            
            # Prepare texts for embedding
            texts = [f"{doc.title}\n{doc.abstract}" for doc in documents]
            
            # Generate embeddings in batches
            embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_embeddings = self._batch_generate_embeddings(batch_texts)
                embeddings.extend(batch_embeddings)
            
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            
            # Update metadata
            for doc, embedding in zip(documents, embeddings):
                doc.embedding = embedding
                self.metadata[doc.id] = doc
            
            # Save index and metadata
            self._save_state()
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_func: Optional[callable] = None
    ) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_func: Optional function to filter results
            
        Returns:
            List of documents with similarity scores
        """
        query_embedding = self._generate_embedding(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search in the index
        distances, indices = self.index.search(query_embedding, k * 2)  # Get more results for filtering
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
                
            doc_ids = list(self.metadata.keys())
            doc = self.metadata[doc_ids[idx]]
            
            if filter_func and not filter_func(doc):
                continue
                
            doc_dict = doc.to_dict()
            doc_dict['score'] = float(1 / (1 + distance))  # Convert distance to similarity score
            results.append(doc_dict)
            
            if len(results) >= k:
                break
        
        return results
    
    def _save_state(self) -> None:
        """Save the index and metadata to disk."""
        try:
            faiss.write_index(self.index, self.index_file)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Error saving vector store state: {e}")
    
    def clear(self) -> None:
        """Clear the index and metadata."""
        with self.lock:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = {}
            self._save_state()

class RAGSystem:
    def __init__(
        self,
        vector_store: VectorStore,
        chunk_size: int = 512,
        chunk_overlap: int = 64
    ):
        """
        Initialize the RAG system.
        
        Args:
            vector_store: VectorStore instance
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
        """
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            
        return chunks
    
    def add_paper(self, paper_data: Dict) -> None:
        """
        Add a paper to the RAG system.
        
        Args:
            paper_data: Dictionary containing paper information
        """
        # Create document object
        doc = Document(
            id=paper_data.get('arxiv_id', str(datetime.now().timestamp())),
            title=paper_data.get('title', ''),
            authors=paper_data.get('authors', ''),
            abstract=paper_data.get('abstract', ''),
            pdf_link=paper_data.get('pdf_link'),
            arxiv_link=paper_data.get('arxiv_link'),
            published=paper_data.get('published', ''),
            categories=paper_data.get('categories', '')
        )
        
        # Add to vector store
        self.vector_store.add_documents([doc])
        
    def query(
        self,
        query: str,
        k: int = 5,
        category_filter: Optional[str] = None,
        date_filter: Optional[tuple] = None
    ) -> List[Dict]:
        """
        Query the RAG system.
        
        Args:
            query: Search query
            k: Number of results to return
            category_filter: Optional category to filter by
            date_filter: Optional tuple of (start_date, end_date) for filtering
            
        Returns:
            List of relevant documents with similarity scores
        """
        def filter_func(doc: Document) -> bool:
            if category_filter and category_filter not in doc.categories:
                return False
                
            if date_filter:
                start_date, end_date = date_filter
                doc_date = datetime.strptime(doc.published[:10], '%Y-%m-%d')
                if not (start_date <= doc_date <= end_date):
                    return False
            
            return True
        
        return self.vector_store.search(query, k=k, filter_func=filter_func if (category_filter or date_filter) else None)

# Usage example:
"""
# Initialize the systems
vector_store = VectorStore()
rag_system = RAGSystem(vector_store)

# Add papers
paper_data = {
    'arxiv_id': '2024.12345',
    'title': 'Example Paper',
    'authors': 'John Doe, Jane Smith',
    'abstract': 'This is an example abstract...',
    'pdf_link': 'https://arxiv.org/pdf/2024.12345.pdf',
    'arxiv_link': 'https://arxiv.org/abs/2024.12345',
    'published': '2024-03-15',
    'categories': 'cs.AI, cs.LG'
}
rag_system.add_paper(paper_data)

# Query the system
results = rag_system.query(
    query="machine learning applications",
    k=5,
    category_filter="cs.AI",
    date_filter=(
        datetime(2024, 1, 1),
        datetime(2024, 12, 31)
    )
)
"""