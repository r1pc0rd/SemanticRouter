"""Alias resolution for upstream MCP servers.

This module provides the AliasResolver class for resolving upstream aliases
to their canonical names. Aliases allow users to refer to upstreams by
friendly names (e.g., "browser" instead of "playwright").
"""

from typing import Dict, List
from ..core.config import RouterConfig


class AliasResolver:
    """Resolves upstream aliases to canonical upstream names.
    
    The AliasResolver builds a mapping from aliases to upstream names from
    the router configuration. It supports case-insensitive resolution and
    provides helpful error messages for unknown aliases.
    
    Attributes:
        _alias_map: Dictionary mapping lowercase aliases to upstream names
        _upstream_names: Set of all upstream names for validation
    """
    
    def __init__(self, config: RouterConfig):
        """Initialize the alias resolver from router configuration.
        
        Args:
            config: Router configuration containing upstream definitions
        """
        self._alias_map: Dict[str, str] = {}
        self._upstream_names: set = set()
        
        # Build alias map from config
        for upstream_name, upstream_config in config.mcp_servers.items():
            self._upstream_names.add(upstream_name)
            
            # Add each alias to the map (case-insensitive)
            for alias in upstream_config.aliases:
                alias_lower = alias.lower()
                if alias_lower in self._alias_map:
                    # Duplicate alias - log warning but allow it (last one wins)
                    pass
                self._alias_map[alias_lower] = upstream_name
    
    def resolve(self, name: str) -> str:
        """Resolve an alias or upstream name to the canonical upstream name.
        
        This method performs case-insensitive resolution. If the name is
        already a valid upstream name, it returns it unchanged. If it's an
        alias, it returns the corresponding upstream name.
        
        Args:
            name: Alias or upstream name to resolve
            
        Returns:
            Canonical upstream name
            
        Raises:
            ValueError: If the name is neither a valid upstream nor a known alias
        """
        # Check if it's already a valid upstream name (case-sensitive)
        if name in self._upstream_names:
            return name
        
        # Try case-insensitive alias lookup
        name_lower = name.lower()
        if name_lower in self._alias_map:
            return self._alias_map[name_lower]
        
        # Not found - provide helpful error message
        available_aliases = sorted(self._alias_map.keys())
        available_upstreams = sorted(self._upstream_names)
        
        error_msg = f"Unknown upstream or alias: '{name}'. "
        if available_aliases:
            error_msg += f"Available aliases: {', '.join(available_aliases)}. "
        error_msg += f"Available upstreams: {', '.join(available_upstreams)}"
        
        raise ValueError(error_msg)
    
    def resolve_multiple(self, names: List[str]) -> List[str]:
        """Resolve multiple aliases/names to canonical upstream names.
        
        This method resolves each name in the input list independently.
        If any name cannot be resolved, a ValueError is raised with details
        about all failed resolutions.
        
        Args:
            names: List of aliases or upstream names to resolve
            
        Returns:
            List of canonical upstream names (same order as input)
            
        Raises:
            ValueError: If any name cannot be resolved
        """
        resolved = []
        errors = []
        
        for name in names:
            try:
                resolved.append(self.resolve(name))
            except ValueError as e:
                errors.append(f"  - {name}: {str(e)}")
        
        if errors:
            error_msg = "Failed to resolve the following names:\n" + "\n".join(errors)
            raise ValueError(error_msg)
        
        return resolved
