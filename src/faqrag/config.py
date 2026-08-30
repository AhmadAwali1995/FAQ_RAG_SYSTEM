"""Application settings, loaded from environment variables and ``.env``.

Every tunable -- model names, endpoints, top-k, thresholds, feature toggles --
lives here so that logic modules stay free of hardcoded values.  Override any
field with an uppercase ``FAQRAG_``-prefixed environment variable, e.g.
``FAQRAG_TOP_K=8`` or ``FAQRAG_RERANK_ENABLED=false``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration for the FAQ RAG system."""

    model_config = SettingsConfigDict(
        env_prefix="FAQRAG_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths -------------------------------------------------------------
    data_path: Path = Field(
        default=PROJECT_ROOT / "data" / "mwfaq_faq_rag.json",
        description="Source FAQ JSON consumed by the indexer.",
    )
    index_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "index",
        description="Directory holding the persisted vector store and BM25 index.",
    )
    log_dir: Path = Field(
        default=PROJECT_ROOT / "logs",
        description="Directory for the per-query retrieval trace log.",
    )

    # --- Embeddings --------------------------------------------------------
    embedding_provider: Literal["ollama", "openai"] = "ollama"
    embedding_model: str = Field(
        default="bge-m3",
        description="Multilingual embedding model. bge-m3 covers Arabic well.",
    )
    embedding_batch_size: int = 16

    # --- Vector store ------------------------------------------------------
    vector_store: Literal["numpy", "chroma"] = Field(
        default="numpy",
        description="Backend behind the VectorStore interface. Swappable by config.",
    )
    collection_name: str = "mwfaq_faq"

    # --- Retrieval ---------------------------------------------------------
    top_k: int = Field(default=5, description="Chunks returned to the generator.")
    candidate_k: int = Field(
        default=20,
        description="Candidates pulled from each retriever before fusion/rerank.",
    )
    fusion_method: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = Field(default=60, description="RRF damping constant.")
    vector_weight: float = 0.6
    lexical_weight: float = 0.4
    language_boost: float = Field(
        default=0.15,
        description="Additive bonus applied to chunks matching the query language.",
    )
    min_relevance_score: float = Field(
        default=0.55,
        description=(
            "Absolute relevance (0-1) the best candidate must clear. Below this "
            "the retriever reports no confident match instead of forcing an "
            "answer. Compared against ScoredChunk.relevance -- the reranker's "
            "rescaled 0-10 judgement, or embedding cosine when reranking is off "
            "-- never against the within-query fusion score."
        ),
    )
    cross_lingual_fallback_score: float = Field(
        default=0.45,
        description=(
            "If the best same-language candidate scores below this, results from "
            "the other language are allowed to compete on equal footing."
        ),
    )

    # --- Reranking ---------------------------------------------------------
    rerank_enabled: bool = Field(
        default=True,
        description="Toggle the reranker off to cut a generation round-trip.",
    )
    rerank_provider: Literal["llm", "none"] = "llm"
    rerank_model: str = Field(
        default="",
        description="Model used for reranking; falls back to llm_model when empty.",
    )
    rerank_candidates: int = Field(default=8, description="Candidates sent to the reranker.")
    rerank_max_tokens: int = Field(
        default=4096,
        description=(
            "Token budget for the rerank call. Scoring many candidates makes a "
            "reasoning model think at length, and a budget exhausted mid-thought "
            "yields an empty completion."
        ),
    )

    # --- Generation --------------------------------------------------------
    llm_provider: Literal["ollama", "openai", "anthropic", "extractive"] = "ollama"
    llm_model: str = "deepseek-v4-flash:cloud"
    answer_style: Literal["saudi", "msa"] = Field(
        default="saudi",
        description=(
            "Voice of generated answers. 'saudi' replies in Saudi dialect with a "
            "human tone; 'msa' keeps formal Modern Standard Arabic. Affects "
            "wording only -- grounding rules are identical for both."
        ),
    )
    llm_temperature: float = 0.0
    llm_max_tokens: int = Field(
        default=2048,
        description=(
            "Token budget per generation. On reasoning models this covers the "
            "hidden reasoning too, so it must leave room for both."
        ),
    )

    # --- Providers ---------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: float = Field(default=180.0, description="Seconds; cloud models can be slow.")
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str = ""

    # --- API ---------------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # --- Async message API (/v1/messages) ----------------------------------
    message_workers: int = Field(
        default=2,
        description=(
            "Concurrent answer workers. The model serialises requests, so a "
            "higher number mostly buys queued threads rather than throughput."
        ),
    )
    message_ttl_seconds: float = Field(
        default=3600.0,
        description="How long a completed message stays collectable before eviction.",
    )
    message_max_stored: int = Field(
        default=1000, description="Cap on retained messages; oldest completed are evicted first."
    )

    # --- Chat UI (optional, fully removable) -------------------------------
    enable_chat_ui: bool = Field(
        default=True,
        description=(
            "Serve the browser chat UI at /chat. Set false to run API-only "
            "without deleting anything; see web/README.md to remove it entirely."
        ),
    )
    web_dir: Path = Field(
        default=PROJECT_ROOT / "web",
        description="Directory holding the chat UI's static files.",
    )

    # --- Logging -----------------------------------------------------------
    log_level: str = "INFO"
    log_retrieval_traces: bool = Field(
        default=True,
        description="Append per-query retrieval scores to logs/retrieval.jsonl.",
    )

    @property
    def effective_rerank_model(self) -> str:
        """Model to use for reranking, defaulting to the generation model."""
        return self.rerank_model or self.llm_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
