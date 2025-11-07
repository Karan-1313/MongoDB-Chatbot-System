#!/usr/bin/env python3
"""Document loading script for the MongoDB Chatbot System.

This script processes PDF and text files from a directory, generates embeddings,
and stores them in MongoDB vector store with progress tracking.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from langchain_core.documents import Document

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.text_processing import DocumentReader, TextProcessor, DocumentChunk
from src.tools.embeddings import EmbeddingGenerator
from src.tools.vector_store import MongoVectorStore
from src.core.config import get_settings, validate_required_env_vars
from src.core.logging import setup_logging
from src.core.exceptions import (
    ChatbotBaseException,
    ValidationError,
    DatabaseError,
    EmbeddingError,
    VectorStoreError,
    ResourceError
)
from src.core.retry import retry_embeddings, retry_mongodb


class DocumentProcessor:
    """Main document processing pipeline."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 50,
        min_chunk_size: int = 100,
        allow_duplicates: bool = False
    ):
        """Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of each text chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            batch_size: Number of documents to process in each batch
            min_chunk_size: Minimum size for a chunk to be considered valid
        """
        self.settings = get_settings()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.min_chunk_size = min_chunk_size
        self.allow_duplicates = allow_duplicates
        
        # Initialize components
        self.document_reader = DocumentReader()
        self.text_processor = TextProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size
        )
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = MongoVectorStore()
        
        self.logger = logging.getLogger(__name__)
        
        # Ensure indexes are created
        try:
            self.vector_store.create_index()
            self.logger.info("Vector store indexes verified/created")
        except Exception as e:
            self.logger.warning(f"Could not create/verify indexes: {e}")
        
        # Statistics tracking
        self.stats = {
            "files_processed": 0,
            "files_failed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "documents_stored": 0,
            "total_processing_time": 0.0,
            "errors": []
        }
        
        self.logger.info(f"Initialized DocumentProcessor with batch_size={batch_size}")
    
    def process_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        clear_existing: bool = False,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Process all supported files in a directory.
        
        Args:
            directory_path: Path to directory containing documents
            recursive: Whether to search subdirectories recursively
            clear_existing: Whether to clear existing documents before loading
            dry_run: If True, only analyze files without processing
            
        Returns:
            Dictionary containing processing statistics
        """
        start_time = time.time()
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise DocumentLoadingError(f"Directory not found: {directory_path}")
        
        if not directory_path.is_dir():
            raise DocumentLoadingError(f"Path is not a directory: {directory_path}")
        
        self.logger.info(f"Starting document processing for: {directory_path}")
        self.logger.info(f"Recursive: {recursive}, Clear existing: {clear_existing}, Dry run: {dry_run}")
        
        # Clear existing documents if requested
        if clear_existing and not dry_run:
            self._clear_existing_documents()
        
        # Find all supported files
        supported_files = self._find_supported_files(directory_path, recursive)
        
        if not supported_files:
            self.logger.warning("No supported files found in directory")
            return self.stats
        
        self.logger.info(f"Found {len(supported_files)} supported files")
        
        if dry_run:
            return self._analyze_files(supported_files)
        
        # Process files in batches
        return self._process_files_in_batches(supported_files)
    
    def _find_supported_files(self, directory_path: Path, recursive: bool) -> List[Path]:
        """Find all supported files in the directory.
        
        Args:
            directory_path: Directory to search
            recursive: Whether to search recursively
            
        Returns:
            List of supported file paths
        """
        supported_extensions = {'.txt', '.pdf'}
        files = []
        
        if recursive:
            pattern = '**/*'
        else:
            pattern = '*'
        
        for file_path in directory_path.glob(pattern):
            if (file_path.is_file() and 
                file_path.suffix.lower() in supported_extensions):
                files.append(file_path)
        
        return sorted(files)
    
    def _analyze_files(self, file_paths: List[Path]) -> Dict[str, Any]:
        """Analyze files without processing them (dry run).
        
        Args:
            file_paths: List of file paths to analyze
            
        Returns:
            Analysis statistics
        """
        analysis = {
            "total_files": len(file_paths),
            "file_types": {},
            "estimated_chunks": 0,
            "total_size_bytes": 0,
            "files_by_type": {}
        }
        
        self.logger.info("Analyzing files (dry run mode)...")
        
        for file_path in tqdm(file_paths, desc="Analyzing files"):
            try:
                # Get file info
                file_size = file_path.stat().st_size
                file_ext = file_path.suffix.lower()
                
                analysis["total_size_bytes"] += file_size
                analysis["file_types"][file_ext] = analysis["file_types"].get(file_ext, 0) + 1
                
                if file_ext not in analysis["files_by_type"]:
                    analysis["files_by_type"][file_ext] = []
                analysis["files_by_type"][file_ext].append(str(file_path))
                
                # Estimate chunks (rough approximation)
                if file_ext == '.txt':
                    # For text files, estimate based on file size
                    estimated_chars = file_size  # Rough approximation
                elif file_ext == '.pdf':
                    # For PDFs, estimate based on typical page content
                    estimated_chars = file_size * 2  # Very rough approximation
                else:
                    estimated_chars = file_size
                
                estimated_chunks = max(1, estimated_chars // self.chunk_size)
                analysis["estimated_chunks"] += estimated_chunks
                
            except Exception as e:
                self.logger.warning(f"Could not analyze file {file_path}: {e}")
                continue
        
        # Log analysis results
        self.logger.info("=== File Analysis Results ===")
        self.logger.info(f"Total files: {analysis['total_files']}")
        self.logger.info(f"Total size: {analysis['total_size_bytes'] / (1024*1024):.2f} MB")
        self.logger.info(f"Estimated chunks: {analysis['estimated_chunks']}")
        self.logger.info(f"File types: {analysis['file_types']}")
        
        return analysis
    
    def _clear_existing_documents(self) -> None:
        """Clear existing documents from the vector store."""
        self.logger.info("Clearing existing documents from vector store...")
        try:
            deleted_count = self.vector_store.clear_collection()
            self.logger.info(f"Cleared {deleted_count} existing documents")
        except Exception as e:
            self.logger.error(f"Failed to clear existing documents: {e}")
            raise DocumentLoadingError(f"Failed to clear existing documents: {e}")
    
    def _process_files_in_batches(self, file_paths: List[Path]) -> Dict[str, Any]:
        """Process files in batches with progress tracking.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Processing statistics
        """
        total_files = len(file_paths)
        processed_files = 0
        
        # Process files in batches
        with tqdm(total=total_files, desc="Processing files") as pbar:
            for i in range(0, total_files, self.batch_size):
                batch_files = file_paths[i:i + self.batch_size]
                
                try:
                    batch_stats = self._process_file_batch(batch_files)
                    processed_files += len(batch_files)
                    
                    # Update progress bar
                    pbar.update(len(batch_files))
                    pbar.set_postfix({
                        'chunks': self.stats['chunks_created'],
                        'stored': self.stats['documents_stored'],
                        'errors': self.stats['files_failed']
                    })
                    
                    # Log batch completion
                    self.logger.debug(f"Completed batch {i//self.batch_size + 1}, processed {processed_files}/{total_files} files")
                    
                except Exception as e:
                    self.logger.error(f"Batch processing failed: {e}")
                    self.stats['errors'].append(f"Batch {i//self.batch_size + 1}: {e}")
                    continue
        
        # Calculate final statistics
        end_time = time.time()
        self.stats['total_processing_time'] = end_time - time.time()
        
        self._log_final_statistics()
        return self.stats
    
    def _process_file_batch(self, file_paths: List[Path]) -> Dict[str, Any]:
        """Process a batch of files.
        
        Args:
            file_paths: List of file paths in the batch
            
        Returns:
            Batch processing statistics
        """
        batch_chunks = []
        batch_documents = []
        
        # Step 1: Read and chunk all files in the batch
        for file_path in file_paths:
            try:
                # Read file content
                content = self.document_reader.read_file(file_path)
                
                # Create chunks
                chunks = self.text_processor.chunk_text(content, str(file_path))
                
                if not chunks:
                    self.logger.warning(f"No chunks created for file: {file_path}")
                    continue
                
                # Convert chunks to LangChain Documents
                for chunk in chunks:
                    doc = Document(
                        page_content=chunk.content,
                        metadata={
                            "source": str(file_path),
                            "chunk_id": chunk.chunk_id,
                            "file_type": file_path.suffix.lower(),
                            "text_length": chunk.text_length,
                            **chunk.metadata
                        }
                    )
                    batch_documents.append(doc)
                    batch_chunks.append(chunk)
                
                self.stats['files_processed'] += 1
                self.stats['chunks_created'] += len(chunks)
                
            except Exception as e:
                self.logger.error(f"Failed to process file {file_path}: {e}")
                self.stats['files_failed'] += 1
                self.stats['errors'].append(f"File {file_path}: {e}")
                continue
        
        if not batch_documents:
            self.logger.warning("No documents created in batch")
            return {}
        
        # Step 2: Generate embeddings for all chunks in batch
        try:
            texts = [doc.page_content for doc in batch_documents]
            embeddings = self.embedding_generator.generate_embeddings_batch(
                texts, 
                batch_size=min(100, len(texts))
            )
            
            self.stats['embeddings_generated'] += len(embeddings)
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings for batch: {e}")
            self.stats['errors'].append(f"Embedding generation: {e}")
            raise DocumentLoadingError(f"Embedding generation failed: {e}")
        
        # Step 3: Store documents with embeddings (with duplicate detection)
        try:
            inserted_ids = self.vector_store.add_documents(
                batch_documents, 
                embeddings, 
                skip_duplicates=not self.allow_duplicates
            )
            self.stats['documents_stored'] += len(inserted_ids)
            
            self.logger.debug(f"Stored {len(inserted_ids)} documents in vector store")
            
        except Exception as e:
            self.logger.error(f"Failed to store documents in batch: {e}")
            self.stats['errors'].append(f"Document storage: {e}")
            raise DocumentLoadingError(f"Document storage failed: {e}")
        
        return {
            "files_in_batch": len(file_paths),
            "chunks_created": len(batch_chunks),
            "documents_stored": len(inserted_ids)
        }
    
    def _log_final_statistics(self) -> None:
        """Log final processing statistics."""
        self.logger.info("=== Document Processing Complete ===")
        self.logger.info(f"Files processed: {self.stats['files_processed']}")
        self.logger.info(f"Files failed: {self.stats['files_failed']}")
        self.logger.info(f"Chunks created: {self.stats['chunks_created']}")
        self.logger.info(f"Embeddings generated: {self.stats['embeddings_generated']}")
        self.logger.info(f"Documents stored: {self.stats['documents_stored']}")
        self.logger.info(f"Total processing time: {self.stats['total_processing_time']:.2f} seconds")
        
        if self.stats['errors']:
            self.logger.warning(f"Errors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:  # Show first 5 errors
                self.logger.warning(f"  - {error}")
            if len(self.stats['errors']) > 5:
                self.logger.warning(f"  ... and {len(self.stats['errors']) - 5} more errors")


def main():
    """Main entry point for the document loading script."""
    parser = argparse.ArgumentParser(
        description="Load documents into MongoDB vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in a directory
  python scripts/load_docs.py /path/to/documents

  # Process with custom settings
  python scripts/load_docs.py /path/to/documents --chunk-size 1500 --batch-size 25

  # Clear existing documents before loading
  python scripts/load_docs.py /path/to/documents --clear-existing

  # Dry run to analyze files without processing
  python scripts/load_docs.py /path/to/documents --dry-run

  # Process only current directory (not recursive)
  python scripts/load_docs.py /path/to/documents --no-recursive
        """
    )
    
    parser.add_argument(
        "directory",
        nargs="?",
        help="Directory containing documents to process"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum size of each text chunk in characters (default: 1000)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Number of characters to overlap between chunks (default: 200)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of documents to process in each batch (default: 50)"
    )
    
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=100,
        help="Minimum size for a chunk to be considered valid (default: 100)"
    )
    
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Clear existing documents before loading new ones"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze files without processing them"
    )
    
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories recursively"
    )
    
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow duplicate documents to be stored"
    )
    
    parser.add_argument(
        "--remove-duplicates",
        action="store_true",
        help="Remove existing duplicate documents before loading"
    )
    
    parser.add_argument(
        "--show-duplicate-stats",
        action="store_true",
        help="Show duplicate statistics without processing files"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate environment variables
        logger.info("Validating configuration...")
        validate_required_env_vars()
        
        # Initialize document processor
        processor = DocumentProcessor(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
            min_chunk_size=args.min_chunk_size,
            allow_duplicates=args.allow_duplicates
        )
        
        # Handle duplicate-related operations
        if args.show_duplicate_stats:
            duplicate_stats = processor.vector_store.get_duplicate_count()
            print("\n=== Duplicate Statistics ===")
            print(f"Total duplicate documents: {duplicate_stats['total_duplicate_documents']}")
            print(f"Unique duplicate groups: {duplicate_stats['unique_duplicate_groups']}")
            if duplicate_stats['duplicate_details']:
                print("\nDuplicate groups:")
                for group in duplicate_stats['duplicate_details'][:5]:  # Show first 5
                    print(f"  - Content hash: {group['_id'][:8]}... (count: {group['count']})")
                    print(f"    Sources: {group['sources']}")
            return
        
        if args.remove_duplicates:
            logger.info("Removing existing duplicate documents...")
            removed_count = processor.vector_store.remove_duplicates()
            print(f"Removed {removed_count} duplicate documents")
            if not args.directory:
                return
        
        # Validate directory argument for processing operations
        if not args.directory and not (args.show_duplicate_stats or args.remove_duplicates):
            parser.error("Directory argument is required unless using --show-duplicate-stats or --remove-duplicates")
        
        # Process directory
        stats = processor.process_directory(
            directory_path=args.directory,
            recursive=not args.no_recursive,
            clear_existing=args.clear_existing,
            dry_run=args.dry_run
        )
        
        # Print summary
        if args.dry_run:
            print("\n=== Analysis Summary ===")
            print(f"Total files found: {stats.get('total_files', 0)}")
            print(f"Estimated chunks: {stats.get('estimated_chunks', 0)}")
            print(f"Total size: {stats.get('total_size_bytes', 0) / (1024*1024):.2f} MB")
            print(f"File types: {stats.get('file_types', {})}")
        else:
            print("\n=== Processing Summary ===")
            print(f"Files processed: {stats['files_processed']}")
            print(f"Files failed: {stats['files_failed']}")
            print(f"Chunks created: {stats['chunks_created']}")
            print(f"Documents stored: {stats['documents_stored']}")
            print(f"Processing time: {stats['total_processing_time']:.2f} seconds")
            
            if stats['files_failed'] > 0:
                print(f"\nWarning: {stats['files_failed']} files failed to process")
                print("Check logs for details")
        
        logger.info("Document loading completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Document loading failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()