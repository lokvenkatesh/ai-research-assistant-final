"""
Document processing for research papers
Handles PDF extraction, chunking, and metadata extraction
"""

import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import hashlib

@dataclass
class DocumentChunk:
    """Represents a chunk of a document"""
    content: str
    metadata: Dict
    chunk_id: str
    page_number: int
    
@dataclass
class Document:
    """Represents a complete document"""
    content: str
    metadata: Dict
    chunks: List[DocumentChunk]
    doc_id: str

class DocumentProcessor:
    """Process research papers (PDFs) for RAG system"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def extract_text_from_pdf(self, pdf_path: Path) -> tuple[str, Dict]:
        """
        Extract text and metadata from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (full_text, metadata)
        """
        try:
            doc = fitz.open(pdf_path)
            
            # Extract metadata
            metadata = {
                'filename': pdf_path.name,
                'path': str(pdf_path),
                'num_pages': len(doc),
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'keywords': doc.metadata.get('keywords', ''),
            }
            
            # Extract text from all pages
            full_text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                full_text += f"\n--- Page {page_num + 1} ---\n"
                full_text += page.get_text()
            
            doc.close()
            
            # Clean the text
            full_text = self._clean_text(full_text)
            
            # Try to extract title if not in metadata
            if not metadata['title']:
                metadata['title'] = self._extract_title_from_text(full_text)
            
            return full_text, metadata
            
        except Exception as e:
            raise Exception(f"Error processing PDF {pdf_path}: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove page headers/footers patterns (common in papers)
        text = re.sub(r'\n\d+\n', '\n', text)  # Page numbers
        
        return text.strip()
    
    def _extract_title_from_text(self, text: str) -> str:
        """Extract title from first lines of text"""
        lines = text.split('\n')[:10]  # Check first 10 lines
        
        for line in lines:
            line = line.strip()
            # Title is usually the first long line
            if len(line) > 20 and len(line) < 200:
                return line
                
        return "Unknown Title"
    
    def chunk_text(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Full document text
            metadata: Document metadata
            
        Returns:
            List of DocumentChunk objects
        """
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_number = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
                
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    metadata={**metadata, 'chunk_number': chunk_number},
                    chunk_id=f"{metadata.get('filename', 'doc')}_{chunk_id}",
                    page_number=self._estimate_page(chunk_text, text, metadata.get('num_pages', 1))
                ))
                
                # Keep overlap
                overlap_size = 0
                overlap_chunk = []
                for s in reversed(current_chunk):
                    overlap_size += len(s.split())
                    overlap_chunk.insert(0, s)
                    if overlap_size >= self.chunk_overlap:
                        break
                
                current_chunk = overlap_chunk
                current_length = overlap_size
                chunk_number += 1
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
            
            chunks.append(DocumentChunk(
                content=chunk_text,
                metadata={**metadata, 'chunk_number': chunk_number},
                chunk_id=f"{metadata.get('filename', 'doc')}_{chunk_id}",
                page_number=self._estimate_page(chunk_text, text, metadata.get('num_pages', 1))
            ))
        
        return chunks
    
    def _estimate_page(self, chunk_text: str, full_text: str, num_pages: int) -> int:
        """Estimate which page a chunk comes from"""
        position = full_text.find(chunk_text[:50])  # Find chunk position
        if position == -1:
            return 1
        
        # Rough estimation based on position
        page = int((position / len(full_text)) * num_pages) + 1
        return min(page, num_pages)
    
    def process_document(self, pdf_path: Path) -> Document:
        """
        Complete document processing pipeline
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Document object with chunks
        """
        print(f"📄 Processing: {pdf_path.name}")
        
        # Extract text and metadata
        full_text, metadata = self.extract_text_from_pdf(pdf_path)
        
        # Create chunks
        chunks = self.chunk_text(full_text, metadata)
        
        # Generate document ID
        doc_id = hashlib.md5(str(pdf_path).encode()).hexdigest()[:12]
        
        print(f"  ✅ Extracted {len(full_text)} characters")
        print(f"  ✅ Created {len(chunks)} chunks")
        
        return Document(
            content=full_text,
            metadata=metadata,
            chunks=chunks,
            doc_id=doc_id
        )
    
    def process_directory(self, directory: Path) -> List[Document]:
        """
        Process all PDFs in a directory
        
        Args:
            directory: Path to directory containing PDFs
            
        Returns:
            List of processed Document objects
        """
        pdf_files = list(directory.glob('*.pdf'))
        
        if not pdf_files:
            print(f"⚠️  No PDF files found in {directory}")
            return []
        
        print(f"\n📚 Processing {len(pdf_files)} PDFs from {directory}")
        
        documents = []
        for pdf_path in pdf_files:
            try:
                doc = self.process_document(pdf_path)
                documents.append(doc)
            except Exception as e:
                print(f"  ❌ Error processing {pdf_path.name}: {str(e)}")
        
        print(f"\n✅ Successfully processed {len(documents)}/{len(pdf_files)} documents")
        
        return documents