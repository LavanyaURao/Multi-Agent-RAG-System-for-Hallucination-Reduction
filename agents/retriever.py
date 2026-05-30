"""
retriever.py
------------
Retriever Agent: runs multiple retrieval strategies over the local knowledge
base and merges their outputs into a single ranked list.
"""

import re

from database.vector_store import VectorStore


class RetrieverAgent:
    def __init__(self, vector_store: VectorStore, knowledge_base_path: str = "data/knowledge_base.txt"):
        self.vector_store = vector_store
        self.knowledge_base_path = knowledge_base_path
        self.keyword_chunks = self._load_keyword_chunks()

    def _load_keyword_chunks(self) -> list[str]:
        with open(self.knowledge_base_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        return [chunk.strip() for chunk in raw.split("\n\n") if chunk.strip()]

    def retrieve(self, query: str, key_terms: list[str] | None = None, top_k: int = 3) -> list[dict]:
        print(f"[RetrieverAgent] Retrieving context for: '{query}'")

        semantic_chunks = self.vector_store.search(query, top_k=top_k)
        keyword_chunks = self._keyword_search(query, key_terms or [], top_k)

        merged: list[dict] = []
        seen_contents = set()

        for rank, chunk in enumerate(semantic_chunks, start=1):
            if chunk not in seen_contents:
                merged.append(
                    {
                        "source": "semantic_faiss",
                        "score": max(top_k - rank + 1, 1),
                        "content": chunk,
                    }
                )
                seen_contents.add(chunk)

        for rank, chunk in enumerate(keyword_chunks, start=1):
            if chunk not in seen_contents:
                merged.append(
                    {
                        "source": "keyword_overlap",
                        "score": max(top_k - rank + 1, 1),
                        "content": chunk,
                    }
                )
                seen_contents.add(chunk)

        print(f"[RetrieverAgent] Retrieved {len(merged)} unique chunks from multiple retrievers.")
        return merged

    def _keyword_search(self, query: str, key_terms: list[str], top_k: int) -> list[str]:
        tokens = self._tokenize(" ".join([query] + key_terms))
        scored: list[tuple[int, str]] = []

        for chunk in self.keyword_chunks:
            overlap = len(tokens & self._tokenize(chunk))
            if overlap > 0:
                scored.append((overlap, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

    def _tokenize(self, text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
