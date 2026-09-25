"""Version-aware indexing: git tracking, incremental updates and lineage."""

from codefusion.versions.git_tracker import CommitInfo, DirectorySource, FileChange, GitTracker
from codefusion.versions.incremental import IncrementalIndexer, UpdateStats, index_path
from codefusion.versions.lineage import LineageEdge, LineageMatcher

__all__ = [
    "CommitInfo",
    "DirectorySource",
    "FileChange",
    "GitTracker",
    "IncrementalIndexer",
    "LineageEdge",
    "LineageMatcher",
    "UpdateStats",
    "index_path",
]
