"""Typed configuration. Every tunable in the pipeline lives here and is driven by YAML."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

INTENTS = (
    "DEFINITION",
    "IMPLEMENTATION",
    "USAGE",
    "CALLER",
    "CALLEE",
    "DATA_FLOW",
    "DEPENDENCY",
    "ERROR",
    "CONFIGURATION",
    "EVOLUTION",
    "GENERAL",
)


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ChunkingConfig(_Base):
    max_chunk_lines: int = 150
    max_module_block_lines: int = 80
    class_skeletons: bool = True
    include_module_blocks: bool = True
    prefer_tree_sitter: bool = True
    include_extensions: list[str] = Field(
        default_factory=lambda: [".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go"]
    )
    exclude_dirs: list[str] = Field(
        default_factory=lambda: [".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".mypy_cache"]
    )
    max_file_bytes: int = 1_000_000


class DenseConfig(_Base):
    enabled: bool = True
    model: str = "jina-code-0.5b"  # registry key or any HF model id
    provider: Literal["auto", "sentence-transformers", "transformers", "hashing"] = "auto"
    device: str = "cpu"
    batch_size: int = 16
    max_seq_length: int = 1024
    top_k: int = 100
    normalize: bool = True
    index_type: Literal["flat", "hnsw", "ivf"] = "flat"
    hnsw_m: int = 32
    ivf_nlist: int = 256
    ivf_nprobe: int = 16
    document_view: Literal["raw", "context+raw", "structural", "raw+structural"] = "raw"
    query_prompt: str | None = None  # override the registry's documented prompt
    document_prompt: str | None = None
    torch_threads: int | None = None
    quantize_int8: bool = False  # dynamic int8 quantisation of Linear layers (CPU speed-up)
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    cache_embeddings: bool = True
    sort_by_length: bool = True
    hashing_dim: int = 768


class BM25Config(_Base):
    enabled: bool = True
    top_k: int = 100
    k1: float = 1.2
    b: float = 0.75
    method: Literal["lucene", "robertson", "atire", "bm25l", "bm25+"] = "lucene"
    identifier_splitting: bool = True
    stemming: bool = True
    remove_stopwords: bool = True
    # Field weights (BM25F-style weighted sum of per-field scores). 0 disables a field.
    fields: dict[str, float] = Field(
        default_factory=lambda: {"raw": 1.0, "identifiers": 0.0, "structural": 0.3, "context": 0.3, "docstring": 0.5}
    )


class SymbolConfig(_Base):
    enabled: bool = True
    top_k: int = 50
    role_weights: dict[str, float] = Field(
        default_factory=lambda: {"def": 3.0, "call": 1.5, "import": 1.2, "attr": 1.0, "ref": 0.5, "param": 0.3}
    )
    # Per-intent multipliers on role weights ("who calls X" wants call sites, not X's definition).
    intent_role_weights: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "CALLER": {"def": 0.2, "call": 2.0},
            # For "what does X call", mentions of X elsewhere are its *callers*: only X's definition
            # is a symbol hit (callees are found by the graph retriever).
            "CALLEE": {"def": 1.5, "call": 0.0, "ref": 0.0, "attr": 0.0, "param": 0.0},
            "DEFINITION": {"def": 1.5, "call": 0.4, "ref": 0.3, "import": 0.5},
            "CONFIGURATION": {"def": 1.5, "call": 0.4, "ref": 0.5, "import": 0.5},
            "USAGE": {"def": 0.4, "call": 1.5, "ref": 1.5, "attr": 1.5, "import": 1.3},
            "DEPENDENCY": {"def": 0.4, "call": 1.3, "ref": 1.2, "attr": 1.3, "import": 2.0},
            # raise sites are calls; "import errors" says nothing about which error is raised
            "ERROR": {"def": 1.2, "call": 1.5, "import": 0.3},
        }
    )
    min_query_symbol_len: int = 3
    scip_index: str | None = None  # optional path to index.scip


class GraphConfig(_Base):
    enabled: bool = True
    backend: Literal["rustworkx", "networkx"] = "rustworkx"
    depth: int = 1
    seeds: int = 10
    top_k: int = 50
    decay: float = 0.5
    max_symbol_fanout: int = 50  # names defined in more places are ambiguous and not expanded; 0 = no limit
    edge_weights: dict[str, float] = Field(
        default_factory=lambda: {"CALLS": 1.0, "REFERENCES": 0.6, "EXTENDS": 0.8, "OVERRIDES": 0.8, "CONTAINS": 0.4, "IMPORTS": 0.3}
    )


class FusionConfig(_Base):
    method: Literal["rrf", "score", "dense_only"] = "rrf"
    rrf_k: int = 60
    query_adaptive: bool = True
    weights: dict[str, float] = Field(default_factory=lambda: {"dense": 1.0, "bm25": 1.0, "symbol": 1.0, "graph": 1.0})
    # Multipliers applied on top of `weights` per detected intent (only when query_adaptive).
    intent_weights: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "DEFINITION": {"dense": 0.6, "bm25": 1.3, "symbol": 2.0, "graph": 0.3},
            "IMPLEMENTATION": {"dense": 1.2, "bm25": 1.0, "symbol": 1.0, "graph": 0.6},
            "USAGE": {"dense": 0.8, "bm25": 1.1, "symbol": 1.6, "graph": 1.2},
            "CALLER": {"dense": 0.6, "bm25": 0.8, "symbol": 1.5, "graph": 2.0},
            "CALLEE": {"dense": 0.6, "bm25": 0.8, "symbol": 1.5, "graph": 2.0},
            "DATA_FLOW": {"dense": 1.3, "bm25": 0.7, "symbol": 1.0, "graph": 1.6},
            "DEPENDENCY": {"dense": 0.8, "bm25": 1.2, "symbol": 1.6, "graph": 1.0},
            "ERROR": {"dense": 1.0, "bm25": 1.3, "symbol": 1.2, "graph": 0.6},
            "CONFIGURATION": {"dense": 0.8, "bm25": 1.3, "symbol": 1.6, "graph": 0.5},
            "EVOLUTION": {"dense": 1.0, "bm25": 1.0, "symbol": 1.2, "graph": 0.8},
            "GENERAL": {"dense": 1.0, "bm25": 1.0, "symbol": 1.0, "graph": 1.0},
        }
    )
    candidate_pool: int = 200


class RerankerConfig(_Base):
    enabled: bool = False
    type: Literal["cross-encoder", "embedding"] = "cross-encoder"
    # ModernBERT cross-encoder trained with code retrieval data; native in transformers (no remote code).
    # jinaai/jina-reranker-v2-base-multilingual needs transformers<5 (its remote code imports removed APIs).
    model: str = "Alibaba-NLP/gte-reranker-modernbert-base"
    candidates: int = 30
    batch_size: int = 16
    max_length: int = 1024
    device: str = "cpu"
    trust_remote_code: bool = False
    # final = alpha * rrf_rank_score + (1 - alpha) * reranker_rank_score  (rank-based, scale free)
    mode: Literal["replace", "interpolate"] = "interpolate"
    alpha: float = 0.3


class StructuralConfig(_Base):
    enabled: bool = True
    defines_symbol: float = 0.6
    calls_symbol: float = 0.3
    references_symbol: float = 0.1
    caller_of_top: float = 0.15
    callee_of_top: float = 0.15
    import_match: float = 0.2
    call_path: float = 0.25
    caller_of_target: float = 0.8  # CALLER / DATA_FLOW intents: candidate calls the queried symbol
    callee_of_target: float = 0.8  # CALLEE / DATA_FLOW intents: queried symbol calls the candidate
    top_n_for_neighbours: int = 3


class DiversityConfig(_Base):
    enabled: bool = True
    dedup_content: bool = True
    dedup_ast: bool = True
    near_duplicate_threshold: float = 0.92  # Jaccard over identifier shingles
    collapse_lineage: bool = True
    mmr: bool = False
    mmr_lambda: float = 0.75


class QueryConfig(_Base):
    classifier: Literal["rules", "none"] = "rules"
    expansion: bool = False
    expansion_max_terms: int = 8
    strip_examples: bool = False  # drop "Examples"/sample I/O sections of long problem statements
    max_query_chars: int = 8000


class VersionConfig(_Base):
    enabled: bool = True
    default_scope: Literal["latest", "all"] = "latest"
    lineage_similarity: float = 0.6
    history_penalty: float = 0.15  # score multiplier penalty for non-latest versions when scope=all


class TrackingConfig(_Base):
    mlflow: bool = False
    experiment: str = "codefusion"
    tracking_uri: str | None = None


class CodeFusionConfig(_Base):
    name: str = "default"
    description: str = ""
    seed: int = 42
    index_dir: str = "artifacts/index"
    cache_dir: str = "artifacts/cache"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    dense: DenseConfig = Field(default_factory=DenseConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    symbols: SymbolConfig = Field(default_factory=SymbolConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    structural: StructuralConfig = Field(default_factory=StructuralConfig)
    diversity: DiversityConfig = Field(default_factory=DiversityConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    versions: VersionConfig = Field(default_factory=VersionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    def enabled_components(self) -> list[str]:
        comps = []
        if self.dense.enabled:
            comps.append(f"dense[{self.dense.model}|{self.dense.document_view}]")
        if self.bm25.enabled:
            comps.append("bm25" + ("+idsplit" if self.bm25.identifier_splitting else ""))
        if self.symbols.enabled:
            comps.append("symbols")
        if self.graph.enabled:
            comps.append(f"graph(d={self.graph.depth})")
        if self.fusion.method != "dense_only":
            comps.append(f"fusion={self.fusion.method}" + ("+adaptive" if self.fusion.query_adaptive else ""))
        if self.reranker.enabled:
            comps.append(f"reranker[{self.reranker.model}]")
        if self.structural.enabled:
            comps.append("structural")
        if self.diversity.enabled:
            comps.append("diversity" + ("+mmr" if self.diversity.mmr else ""))
        if self.query.expansion:
            comps.append("query_expansion")
        return comps
