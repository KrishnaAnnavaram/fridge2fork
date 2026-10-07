"""Settings from environment variables and a local ``.env`` file. Every value has an offline default."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def read_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Read ``KEY=VALUE`` lines. Empty values are skipped. Real environment variables win."""
    path = Path(path)
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                out[k.strip()] = v.strip()
    return out


@dataclass(frozen=True)
class Settings:
    recipes_path: Path = Path("data/recipes.csv")
    index_dir: Path = Path("artifacts/index")
    embedder: str = "svd"  # svd (offline, scikit-learn) or word2vec (gensim extra)
    vector_index: str = "numpy"  # numpy (exact cosine) or faiss (faiss extra)
    dim: int = 64
    seed: int = 42
    min_count: int = 2
    dense_k: int = 100
    overlap_k: int = 200
    top_k: int = 5
    max_missing: int | None = None
    assume_pantry: bool = True
    llm: str = "none"  # none or openai
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    llm_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if self.embedder not in ("svd", "word2vec"):
            raise ValueError("FRIDGE2FORK_EMBEDDER must be svd or word2vec")
        if self.vector_index not in ("numpy", "faiss"):
            raise ValueError("FRIDGE2FORK_VECTOR_INDEX must be numpy or faiss")
        if self.llm not in ("none", "openai"):
            raise ValueError("FRIDGE2FORK_LLM must be none or openai")
        if not 2 <= self.dim <= 512:
            raise ValueError("dim must be between 2 and 512")
        if min(self.dense_k, self.overlap_k, self.top_k, self.min_count) < 1:
            raise ValueError("dense_k, overlap_k, top_k and min_count must be at least 1")
        if self.max_missing is not None and self.max_missing < 0:
            raise ValueError("max_missing must not be negative")

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        e = {**read_dotenv(), **os.environ} if env is None else env
        d = cls()
        mm = e.get("FRIDGE2FORK_MAX_MISSING")
        return cls(
            recipes_path=Path(e.get("FRIDGE2FORK_RECIPES", str(d.recipes_path))),
            index_dir=Path(e.get("FRIDGE2FORK_INDEX_DIR", str(d.index_dir))),
            embedder=e.get("FRIDGE2FORK_EMBEDDER", d.embedder),
            vector_index=e.get("FRIDGE2FORK_VECTOR_INDEX", d.vector_index),
            dim=int(e.get("FRIDGE2FORK_DIM", d.dim)),
            seed=int(e.get("FRIDGE2FORK_SEED", d.seed)),
            min_count=int(e.get("FRIDGE2FORK_MIN_COUNT", d.min_count)),
            dense_k=int(e.get("FRIDGE2FORK_DENSE_K", d.dense_k)),
            overlap_k=int(e.get("FRIDGE2FORK_OVERLAP_K", d.overlap_k)),
            top_k=int(e.get("FRIDGE2FORK_TOP_K", d.top_k)),
            max_missing=int(mm) if mm else None,
            assume_pantry=e.get("FRIDGE2FORK_ASSUME_PANTRY", "true").lower() in ("1", "true", "yes"),
            llm=e.get("FRIDGE2FORK_LLM", d.llm),
            llm_model=e.get("FRIDGE2FORK_LLM_MODEL", d.llm_model),
            llm_base_url=e.get("FRIDGE2FORK_LLM_BASE_URL") or None,
            llm_timeout_s=float(e.get("FRIDGE2FORK_LLM_TIMEOUT_S", d.llm_timeout_s)),
        )
