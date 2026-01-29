"""
Embedding engine for generating semantic embeddings using sentence-transformers.

This module provides the EmbeddingEngine class that loads and manages the
all-MiniLM-L6-v2 model for generating 384-dimensional embeddings.
"""
import asyncio
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from mcp_router.core.models import ToolMetadata
from mcp_router.embedding.utils import generate_tool_embedding_text


class EmbeddingEngine:
    """
    Manages embedding generation using sentence-transformers.
    
    Uses the all-MiniLM-L6-v2 model which produces 384-dimensional embeddings.
    The model is loaded locally and runs without external API calls.
    """
    
    MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
    EMBEDDING_DIMENSION = 384
    
    def __init__(self) -> None:
        """Initialize the embedding engine (model not loaded yet)."""
        self._model: Optional[SentenceTransformer] = None
    
    async def initialize(self) -> None:
        """
        Load the sentence-transformers model asynchronously.
        
        Downloads the model if not cached locally. This is an async operation
        to avoid blocking during model loading.
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(self.MODEL_NAME)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}") from e
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Input text to embed
            
        Returns:
            384-dimensional numpy array
            
        Raises:
            RuntimeError: If model not initialized
            ValueError: If text is empty
        """
        if self._model is None:
            raise RuntimeError("Embedding engine not initialized. Call initialize() first.")
        
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")
        
        # Generate embedding (returns numpy array)
        embedding = self._model.encode(text, convert_to_numpy=True)
        
        # Ensure correct shape
        if embedding.shape != (self.EMBEDDING_DIMENSION,):
            raise RuntimeError(
                f"Unexpected embedding dimension: {embedding.shape}, "
                f"expected ({self.EMBEDDING_DIMENSION},)"
            )
        
        return embedding
    
    def generate_embeddings_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Generate embeddings for multiple texts in a batch.
        
        More efficient than calling generate_embedding() multiple times.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of 384-dimensional numpy arrays
            
        Raises:
            RuntimeError: If model not initialized
            ValueError: If any text is empty
        """
        if self._model is None:
            raise RuntimeError("Embedding engine not initialized. Call initialize() first.")
        
        if not texts:
            return []
        
        # Validate all texts are non-empty
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Cannot generate embedding for empty text at index {i}")
        
        # Generate embeddings in batch (more efficient)
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        
        # Convert to list of individual arrays
        return [embeddings[i] for i in range(len(texts))]
    
    @property
    def is_initialized(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._model is not None
    
    async def generate_tool_embeddings(self, tools: list[ToolMetadata]) -> None:
        """
        Generate embeddings for all tools and cache them in the tool objects.
        
        This method generates embedding text for each tool, creates embeddings
        in batch for efficiency, and stores them in the tool's embedding field.
        
        Args:
            tools: List of tool metadata objects to generate embeddings for
            
        Raises:
            RuntimeError: If model not initialized
            ValueError: If any tool generates empty embedding text
        """
        if not tools:
            return
        
        # Generate embedding text for all tools
        embedding_texts = [generate_tool_embedding_text(tool) for tool in tools]
        
        # Generate embeddings in batch (more efficient)
        embeddings = self.generate_embeddings_batch(embedding_texts)
        
        # Store embeddings in tool objects
        for tool, embedding in zip(tools, embeddings):
            tool.embedding = embedding
