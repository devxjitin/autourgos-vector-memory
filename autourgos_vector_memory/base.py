"""
base.py — Re-exports BaseRetriever and Document from autourgos-memory, the
package that owns these interfaces, to avoid divergent duplicate copies
across the memory-family packages.
"""
from autourgos_memory import BaseRetriever, Document

__all__ = ["BaseRetriever", "Document"]
