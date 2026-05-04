"""
Tag Registry - Unique sector tag allocation.

This module manages the allocation of unique sector tags for
use with ACS scripts and level triggers.
"""

from __future__ import annotations

from typing import Optional


class TagRegistry:
    """
    Registry for unique sector tags.

    Tags in Doom are used to identify sectors for scripting purposes
    (doors, lifts, teleporters, etc.). This registry ensures tags
    are unique and can be reserved or named for easy reference.

    Attributes:
        _next_tag: Next available tag number.
        _used_tags: Set of all used tag numbers.
        _named_tags: Mapping of names to tag numbers.
    """

    def __init__(self, start_tag: int = 1):
        """
        Initialize the tag registry.

        Args:
            start_tag: First tag number to allocate.
        """
        self._next_tag = start_tag
        self._used_tags: set[int] = set()
        self._named_tags: dict[str, int] = {}

    def allocate(self, name: Optional[str] = None) -> int:
        """
        Allocate a new unique tag.

        Args:
            name: Optional name for the tag (for later lookup).

        Returns:
            The allocated tag number.
        """
        tag = self._next_tag
        self._next_tag += 1
        self._used_tags.add(tag)

        if name is not None:
            self._named_tags[name] = tag

        return tag

    def reserve(self, tag: int, name: Optional[str] = None) -> int:
        """
        Reserve a specific tag number.

        Args:
            tag: The specific tag to reserve.
            name: Optional name for the tag.

        Returns:
            The reserved tag number.

        Raises:
            ValueError: If the tag is already in use.
        """
        if tag in self._used_tags:
            raise ValueError(f"Tag {tag} is already in use")

        self._used_tags.add(tag)

        if name is not None:
            self._named_tags[name] = tag

        # Update next_tag if necessary
        if tag >= self._next_tag:
            self._next_tag = tag + 1

        return tag

    def get(self, name: str) -> int:
        """
        Get a tag by its name.

        Args:
            name: The name of the tag.

        Returns:
            The tag number.

        Raises:
            KeyError: If no tag with that name exists.
        """
        if name not in self._named_tags:
            raise KeyError(f"No tag named '{name}'")
        return self._named_tags[name]

    def get_or_allocate(self, name: str) -> int:
        """
        Get a tag by name, allocating a new one if it doesn't exist.

        Args:
            name: The name of the tag.

        Returns:
            The tag number (existing or newly allocated).
        """
        if name in self._named_tags:
            return self._named_tags[name]
        return self.allocate(name)

    def is_used(self, tag: int) -> bool:
        """
        Check if a tag number is already used.

        Args:
            tag: The tag number to check.

        Returns:
            True if the tag is in use.
        """
        return tag in self._used_tags

    def list_named_tags(self) -> dict[str, int]:
        """
        Get all named tags.

        Returns:
            Dictionary mapping names to tag numbers.
        """
        return self._named_tags.copy()

    def reset(self, start_tag: int = 1) -> None:
        """
        Reset the registry to initial state.

        Args:
            start_tag: First tag number to allocate.
        """
        self._next_tag = start_tag
        self._used_tags.clear()
        self._named_tags.clear()
