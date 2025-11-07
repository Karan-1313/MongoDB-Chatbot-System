"""Text processing utilities for the MongoDB Chatbot System."""

import logging
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import PyPDF2
from io import BytesIO

logger = logging.getLogger(__name__)


class TextProcessingError(Exception):
    """Custom exception for text processing errors."""
    pass


class DocumentChunk:
    """Represents a chunk of text from a document."""
    
    def __init__(
        self, 
        content: str, 
        chunk_id: int, 
        source: str, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize a document chunk.
        
        Args:
            content: The text content of the chunk.
            chunk_id: Unique identifier for the chunk within the document.
            source: Source file path or identifier.
            metadata: Additional metadata for the chunk.
        """
        self.content = content
        self.chunk_id = chunk_id
        self.source = source
        self.metadata = metadata or {}
        self.text_length = len(content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary representation."""
        return {
            "content": self.content,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "metadata": self.metadata,
            "text_length": self.text_length
        }
    
    def __repr__(self) -> str:
        return f"DocumentChunk(chunk_id={self.chunk_id}, source='{self.source}', length={self.text_length})"


class TextProcessor:
    """Text processing utilities for document chunking and cleaning."""
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        min_chunk_size: int = 100
    ):
        """Initialize the text processor.
        
        Args:
            chunk_size: Maximum size of each text chunk in characters.
            chunk_overlap: Number of characters to overlap between chunks.
            min_chunk_size: Minimum size for a chunk to be considered valid.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        logger.info(
            f"Initialized TextProcessor with chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}, min_size={min_chunk_size}"
        )
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text content.
        
        Args:
            text: Raw text to clean.
            
        Returns:
            Cleaned text.
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/]', '', text)
        
        # Remove multiple consecutive punctuation marks
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def chunk_text(self, text: str, source: str = "") -> List[DocumentChunk]:
        """Split text into overlapping chunks.
        
        Args:
            text: Text to chunk.
            source: Source identifier for the text.
            
        Returns:
            List of DocumentChunk objects.
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            logger.warning(f"Text too short to chunk: {len(text) if text else 0} characters")
            return []
        
        # Clean the text first
        cleaned_text = self.clean_text(text)
        
        if len(cleaned_text) < self.min_chunk_size:
            logger.warning(f"Cleaned text too short: {len(cleaned_text)} characters")
            return []
        
        chunks = []
        chunk_id = 0
        start = 0
        
        while start < len(cleaned_text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at a sentence boundary
            if end < len(cleaned_text):
                # Look for sentence endings within the last 200 characters
                search_start = max(start + self.chunk_size - 200, start)
                sentence_end = self._find_sentence_boundary(cleaned_text, search_start, end)
                
                if sentence_end > start:
                    end = sentence_end
            
            # Extract chunk content
            chunk_content = cleaned_text[start:end].strip()
            
            # Only create chunk if it meets minimum size requirement
            if len(chunk_content) >= self.min_chunk_size:
                chunk = DocumentChunk(
                    content=chunk_content,
                    chunk_id=chunk_id,
                    source=source,
                    metadata={
                        "start_pos": start,
                        "end_pos": end,
                        "original_length": len(text)
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # Move start position for next chunk (with overlap)
            start = max(end - self.chunk_overlap, start + 1)
            
            # Prevent infinite loop
            if start >= len(cleaned_text):
                break
        
        logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find the best sentence boundary within a range.
        
        Args:
            text: Text to search in.
            start: Start position to search from.
            end: End position to search to.
            
        Returns:
            Position of sentence boundary, or end if none found.
        """
        # Look for sentence endings (., !, ?) followed by whitespace
        sentence_pattern = r'[.!?]\s+'
        
        # Search backwards from end position
        for match in re.finditer(sentence_pattern, text[start:end]):
            boundary = start + match.end()
            if boundary > start + self.min_chunk_size:
                return boundary
        
        # If no sentence boundary found, look for paragraph breaks
        paragraph_pattern = r'\n\s*\n'
        for match in re.finditer(paragraph_pattern, text[start:end]):
            boundary = start + match.end()
            if boundary > start + self.min_chunk_size:
                return boundary
        
        # If no good boundary found, return original end
        return end


class DocumentReader:
    """Document reader for various file formats."""
    
    def __init__(self):
        """Initialize the document reader."""
        self.supported_extensions = {'.txt', '.pdf'}
        logger.info(f"Initialized DocumentReader supporting: {self.supported_extensions}")
    
    def read_file(self, file_path: Union[str, Path]) -> str:
        """Read content from a file.
        
        Args:
            file_path: Path to the file to read.
            
        Returns:
            Text content of the file.
            
        Raises:
            TextProcessingError: If file reading fails.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise TextProcessingError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise TextProcessingError(f"Path is not a file: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension not in self.supported_extensions:
            raise TextProcessingError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {self.supported_extensions}"
            )
        
        try:
            if extension == '.txt':
                return self._read_text_file(file_path)
            elif extension == '.pdf':
                return self._read_pdf_file(file_path)
            else:
                raise TextProcessingError(f"No reader implemented for {extension}")
                
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            raise TextProcessingError(f"Failed to read file {file_path}: {e}")
    
    def _read_text_file(self, file_path: Path) -> str:
        """Read content from a text file.
        
        Args:
            file_path: Path to the text file.
            
        Returns:
            Text content.
        """
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                    logger.debug(f"Successfully read text file with {encoding} encoding")
                    return content
            except UnicodeDecodeError:
                continue
        
        raise TextProcessingError(f"Could not decode text file with any supported encoding")
    
    def _read_pdf_file(self, file_path: Path) -> str:
        """Read content from a PDF file.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            Extracted text content.
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if len(pdf_reader.pages) == 0:
                    raise TextProcessingError("PDF file has no pages")
                
                text_content = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(page_text)
                        logger.debug(f"Extracted text from page {page_num + 1}")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        continue
                
                if not text_content:
                    raise TextProcessingError("No text could be extracted from PDF")
                
                full_text = '\n\n'.join(text_content)
                logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
                
                return full_text
                
        except Exception as e:
            if isinstance(e, TextProcessingError):
                raise
            raise TextProcessingError(f"Failed to read PDF file: {e}")
    
    def read_directory(self, directory_path: Union[str, Path]) -> Dict[str, str]:
        """Read all supported files from a directory.
        
        Args:
            directory_path: Path to the directory.
            
        Returns:
            Dictionary mapping file paths to their content.
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise TextProcessingError(f"Directory not found: {directory_path}")
        
        if not directory_path.is_dir():
            raise TextProcessingError(f"Path is not a directory: {directory_path}")
        
        file_contents = {}
        processed_count = 0
        error_count = 0
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    content = self.read_file(file_path)
                    file_contents[str(file_path)] = content
                    processed_count += 1
                    logger.debug(f"Successfully read: {file_path}")
                except TextProcessingError as e:
                    logger.error(f"Failed to read {file_path}: {e}")
                    error_count += 1
                    continue
        
        logger.info(
            f"Read {processed_count} files from directory. "
            f"Errors: {error_count}"
        )
        
        return file_contents


def create_text_processor(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    min_chunk_size: int = 100
) -> TextProcessor:
    """Factory function to create a TextProcessor instance.
    
    Args:
        chunk_size: Maximum size of each text chunk.
        chunk_overlap: Number of characters to overlap between chunks.
        min_chunk_size: Minimum size for a valid chunk.
        
    Returns:
        Configured TextProcessor instance.
    """
    return TextProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size
    )


def create_document_reader() -> DocumentReader:
    """Factory function to create a DocumentReader instance.
    
    Returns:
        DocumentReader instance.
    """
    return DocumentReader()