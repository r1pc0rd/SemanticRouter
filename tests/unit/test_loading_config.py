"""
Unit tests for LoadingConfig dataclass.
"""
import pytest

from mcp_router.core.config import LoadingConfig


class TestLoadingConfigDefaults:
    """Test default values for LoadingConfig."""
    
    def test_default_values(self):
        """Test that LoadingConfig has correct default values."""
        config = LoadingConfig()
        
        assert config.auto_load == ["all"]
        assert config.lazy_load is True
        assert config.cache_embeddings is True
        assert config.connection_timeout == 30
        assert config.max_concurrent_upstreams == 10
        assert config.rate_limit == 5
    
    def test_custom_values(self):
        """Test that LoadingConfig accepts custom values."""
        config = LoadingConfig(
            auto_load=["upstream1", "upstream2"],
            lazy_load=False,
            cache_embeddings=False,
            connection_timeout=60,
            max_concurrent_upstreams=5,
            rate_limit=10
        )
        
        assert config.auto_load == ["upstream1", "upstream2"]
        assert config.lazy_load is False
        assert config.cache_embeddings is False
        assert config.connection_timeout == 60
        assert config.max_concurrent_upstreams == 5
        assert config.rate_limit == 10


class TestLoadingConfigValidation:
    """Test validation logic for LoadingConfig."""
    
    def test_negative_connection_timeout(self):
        """Test that negative connection_timeout raises ValueError."""
        with pytest.raises(ValueError, match="connection_timeout must be positive"):
            LoadingConfig(connection_timeout=-1)
    
    def test_zero_connection_timeout(self):
        """Test that zero connection_timeout raises ValueError."""
        with pytest.raises(ValueError, match="connection_timeout must be positive"):
            LoadingConfig(connection_timeout=0)
    
    def test_negative_max_concurrent_upstreams(self):
        """Test that negative max_concurrent_upstreams raises ValueError."""
        with pytest.raises(ValueError, match="max_concurrent_upstreams must be positive"):
            LoadingConfig(max_concurrent_upstreams=-1)
    
    def test_zero_max_concurrent_upstreams(self):
        """Test that zero max_concurrent_upstreams raises ValueError."""
        with pytest.raises(ValueError, match="max_concurrent_upstreams must be positive"):
            LoadingConfig(max_concurrent_upstreams=0)
    
    def test_negative_rate_limit(self):
        """Test that negative rate_limit raises ValueError."""
        with pytest.raises(ValueError, match="rate_limit must be positive"):
            LoadingConfig(rate_limit=-1)
    
    def test_zero_rate_limit(self):
        """Test that zero rate_limit raises ValueError."""
        with pytest.raises(ValueError, match="rate_limit must be positive"):
            LoadingConfig(rate_limit=0)
    
    def test_auto_load_not_list(self):
        """Test that non-list auto_load raises ValueError."""
        with pytest.raises(ValueError, match="auto_load must be a list"):
            LoadingConfig(auto_load="all")  # type: ignore
    
    def test_empty_auto_load(self):
        """Test that empty auto_load list is valid."""
        config = LoadingConfig(auto_load=[])
        assert config.auto_load == []
    
    def test_auto_load_with_all(self):
        """Test that auto_load with 'all' is valid."""
        config = LoadingConfig(auto_load=["all"])
        assert config.auto_load == ["all"]
    
    def test_auto_load_with_specific_upstreams(self):
        """Test that auto_load with specific upstream names is valid."""
        config = LoadingConfig(auto_load=["upstream1", "upstream2", "upstream3"])
        assert config.auto_load == ["upstream1", "upstream2", "upstream3"]


class TestLoadingConfigEdgeCases:
    """Test edge cases for LoadingConfig."""
    
    def test_very_large_timeout(self):
        """Test that very large timeout values are accepted."""
        config = LoadingConfig(connection_timeout=3600)
        assert config.connection_timeout == 3600
    
    def test_very_large_max_concurrent(self):
        """Test that very large max_concurrent_upstreams values are accepted."""
        config = LoadingConfig(max_concurrent_upstreams=1000)
        assert config.max_concurrent_upstreams == 1000
    
    def test_very_large_rate_limit(self):
        """Test that very large rate_limit values are accepted."""
        config = LoadingConfig(rate_limit=1000)
        assert config.rate_limit == 1000
    
    def test_minimum_valid_values(self):
        """Test that minimum valid values (1) are accepted."""
        config = LoadingConfig(
            connection_timeout=1,
            max_concurrent_upstreams=1,
            rate_limit=1
        )
        assert config.connection_timeout == 1
        assert config.max_concurrent_upstreams == 1
        assert config.rate_limit == 1
