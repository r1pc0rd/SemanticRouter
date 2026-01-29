"""
Cosine similarity computation for semantic search.

This module provides functions to compute similarity between embedding vectors.
"""
import numpy as np


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Cosine similarity measures the cosine of the angle between two vectors,
    ranging from -1 (opposite) to 1 (identical direction).
    
    Args:
        vec_a: First vector
        vec_b: Second vector
        
    Returns:
        Cosine similarity score between -1 and 1
        
    Raises:
        ValueError: If vectors have different dimensions
        
    Examples:
        >>> import numpy as np
        >>> vec1 = np.array([1.0, 0.0, 0.0])
        >>> vec2 = np.array([1.0, 0.0, 0.0])
        >>> cosine_similarity(vec1, vec2)
        1.0
        
        >>> vec3 = np.array([0.0, 1.0, 0.0])
        >>> cosine_similarity(vec1, vec3)
        0.0
    """
    if vec_a.shape != vec_b.shape:
        raise ValueError(
            f'Vectors must have same dimensionality. '
            f'Got {vec_a.shape} and {vec_b.shape}'
        )
    
    # Compute dot product
    dot_product = np.dot(vec_a, vec_b)
    
    # Compute norms
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    # Handle zero vectors
    denominator = norm_a * norm_b
    if denominator == 0:
        return 0.0
    
    # Compute cosine similarity
    similarity = dot_product / denominator
    
    return float(similarity)


def compute_similarities(query_embedding: np.ndarray, tool_embeddings: list[np.ndarray]) -> list[float]:
    """
    Compute cosine similarities between a query and multiple tool embeddings.
    
    Args:
        query_embedding: Query embedding vector
        tool_embeddings: List of tool embedding vectors
        
    Returns:
        List of similarity scores
        
    Raises:
        ValueError: If any tool embedding has different dimensions than query
    """
    similarities = []
    
    for tool_embedding in tool_embeddings:
        similarity = cosine_similarity(query_embedding, tool_embedding)
        similarities.append(similarity)
    
    return similarities
