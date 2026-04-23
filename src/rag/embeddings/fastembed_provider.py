"""fastembed provider — production embeddings with an ~80MB ONNX model.

Not imported at module load to keep CI lightweight. The class only imports
`fastembed` when instantiated.
"""

from __future__ import annotations


class FastEmbedProvider:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dim: int = 384) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:  # pragma: no cover - only in real envs
            raise RuntimeError(
                "fastembed not installed. `pip install fastembed` to use this provider."
            ) from e
        self._model = TextEmbedding(model_name=model_name)
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        return [list(v) for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:  # pragma: no cover
        return list(next(self._model.embed([text])))
