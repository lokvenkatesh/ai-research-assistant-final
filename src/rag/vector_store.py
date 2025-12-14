"""
Vector store implementation using FAISS
Handles document embedding and similarity search
"""
import sys
from pathlib import Path
# Add this line:
sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss
import numpy as np

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .document_processor import Document, DocumentChunk

class VectorStore:
    """Vector database for semantic search"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize vector store
        
        Args:
            embedding_model: Name of the sentence-transformers model
        """
        print(f"🔧 Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index (using L2 distance)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Store metadata for each vector
        self.chunk_metadata = []
        self.document_map = {}  # Map doc_id to document metadata
        
        print(f"  ✅ Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Create embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.embedding_model.encode(text, convert_to_numpy=True)
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Create embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            
        Returns:
            Matrix of embedding vectors
        """
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to the vector store
        
        Args:
            documents: List of Document objects to add
        """
        print(f"\n📥 Adding {len(documents)} documents to vector store")
        
        all_chunks = []
        chunk_texts = []
        
        # Collect all chunks and their metadata
        for doc in documents:
            self.document_map[doc.doc_id] = doc.metadata
            
            for chunk in doc.chunks:
                all_chunks.append(chunk)
                chunk_texts.append(chunk.content)
        
        print(f"  📊 Total chunks to embed: {len(all_chunks)}")
        
        # Create embeddings
        print("  🔄 Creating embeddings...")
        embeddings = self.embed_texts(chunk_texts)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.chunk_metadata.extend(all_chunks)
        
        print(f"  ✅ Added {len(all_chunks)} chunks to vector store")
        print(f"  📈 Total vectors in store: {self.index.ntotal}")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            print("⚠️  Vector store is empty!")
            return []
        
        # Embed query
        query_embedding = self.embed_text(query).astype('float32').reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunk_metadata):
                chunk = self.chunk_metadata[idx]
                # Convert L2 distance to similarity score (inverse)
                similarity = 1 / (1 + distance)
                results.append((chunk, float(similarity)))
        
        return results
    
    def search_with_metadata_filter(
        self, 
        query: str, 
        top_k: int = 5,
        metadata_filter: Dict = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search with metadata filtering
        
        Args:
            query: Search query
            top_k: Number of results to return
            metadata_filter: Dict of metadata key-value pairs to filter by
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        # Get initial results
        initial_results = self.search(query, top_k * 3)  # Get more results
        
        # Filter by metadata
        if metadata_filter:
            filtered_results = []
            for chunk, score in initial_results:
                match = True
                for key, value in metadata_filter.items():
                    if key not in chunk.metadata or chunk.metadata[key] != value:
                        match = False
                        break
                if match:
                    filtered_results.append((chunk, score))
            
            return filtered_results[:top_k]
        
        return initial_results[:top_k]
    
    def save(self, save_path: Path):
        """
        Save vector store to disk
        
        Args:
            save_path: Directory to save to
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path / "index.faiss"))
        
        # Save metadata
        with open(save_path / "metadata.pkl", 'wb') as f:
            pickle.dump({
                'chunk_metadata': self.chunk_metadata,
                'document_map': self.document_map,
                'embedding_dim': self.embedding_dim
            }, f)
        
        print(f"💾 Vector store saved to {save_path}")
    
    def load(self, load_path: Path):
        """
        Load vector store from disk
        
        Args:
            load_path: Directory to load from
        """
        load_path = Path(load_path)
        
        if not (load_path / "index.faiss").exists():
            raise FileNotFoundError(f"No vector store found at {load_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(load_path / "index.faiss"))
        
        # Load metadata
        with open(load_path / "metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.chunk_metadata = data['chunk_metadata']
            self.document_map = data['document_map']
            self.embedding_dim = data['embedding_dim']
        
        print(f"📂 Vector store loaded from {load_path}")
        print(f"  📊 Total vectors: {self.index.ntotal}")
        print(f"  📚 Documents: {len(self.document_map)}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            'total_vectors': self.index.ntotal,
            'total_documents': len(self.document_map),
            'embedding_dimension': self.embedding_dim,
            'total_chunks': len(self.chunk_metadata)
        }