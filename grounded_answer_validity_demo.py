"""
Grounded answer validity with local and external evidence.

External evidence uses Wikipedia search + summary endpoints so the app can
validate open-domain answers against public sources without additional API keys.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any

from dotenv import load_dotenv
from groq import Groq


USER_AGENT = "GenAI-EAGV-Demo/1.0"


def _safe_json_load(text: str, fallback: dict[str, Any]) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return fallback
        return fallback


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def _fetch_json(url: str) -> dict[str, Any] | None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _dedupe_evidence(items: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for item in items:
        key = (_normalize_text(item.get("title", "")), _normalize_text(item.get("content", "")))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        deduped.append(item)

    return deduped


def _extract_claim_tokens(text: str) -> list[str]:
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "to",
        "and", "for", "with", "that", "this", "it", "as", "by", "from", "at",
        "be", "or", "if", "but", "its", "their", "his", "her",
    }
    return [token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if token not in stopwords]


class LocalKnowledgeRetriever:
    def __init__(self, knowledge_base_path: str = "data/knowledge_base.txt"):
        self.knowledge_base_path = knowledge_base_path
        self.chunks = self._load_chunks()

    def _load_chunks(self) -> list[str]:
        with open(self.knowledge_base_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        return [chunk.strip() for chunk in raw.split("\n\n") if chunk.strip()]

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, str]]:
        query_tokens = _tokenize(query)
        scored: list[tuple[int, str]] = []

        for chunk in self.chunks:
            chunk_tokens = _tokenize(chunk)
            overlap = len(query_tokens & chunk_tokens)
            scored.append((overlap, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_chunks = [chunk for score, chunk in scored[:top_k] if score > 0]

        if not top_chunks:
            top_chunks = self.chunks[:top_k]

        return [
            {
                "source_type": "local_kb",
                "title": f"Local KB Chunk {idx + 1}",
                "url": self.knowledge_base_path,
                "content": chunk,
            }
            for idx, chunk in enumerate(top_chunks)
        ]


class ExternalKnowledgeRetriever:
    def __init__(self, top_k: int = 2):
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict[str, str]]:
        encoded_query = urllib.parse.quote(query)
        search_url = (
            "https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={encoded_query}&utf8=1&format=json&srlimit={self.top_k}"
        )
        search_response = _fetch_json(search_url)
        if not search_response:
            return []

        search_results = search_response.get("query", {}).get("search", [])
        evidence: list[dict[str, str]] = []

        for result in search_results:
            title = result.get("title", "").strip()
            if not title:
                continue

            summary_url = (
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                + urllib.parse.quote(title, safe="")
            )
            summary_response = _fetch_json(summary_url)
            if not summary_response:
                continue

            extract = (summary_response.get("extract") or "").strip()
            content_url = (
                summary_response.get("content_urls", {})
                .get("desktop", {})
                .get("page")
            ) or f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"

            if extract:
                evidence.append(
                    {
                        "source_type": "external_web",
                        "title": title,
                        "url": content_url,
                        "content": extract,
                    }
                )

        return evidence


class GroundedAnswerValidityPipeline:
    def __init__(
        self,
        api_key: str,
        knowledge_base_path: str = "data/knowledge_base.txt",
        retriever_top_k: int = 3,
        external_top_k: int = 2,
    ):
        self.client = Groq(api_key=api_key)
        self.retriever_top_k = retriever_top_k
        self.local_retriever = LocalKnowledgeRetriever(knowledge_base_path=knowledge_base_path)
        self.external_retriever = ExternalKnowledgeRetriever(top_k=external_top_k)
        self.answer_model = "llama-3.3-70b-versatile"
        self.check_model = "llama-3.1-8b-instant"

    def _chat(self, model: str, system_prompt: str, user_prompt: str, max_tokens: int = 700) -> str:
        response = self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def _select_necessary_evidence(
        self,
        query: str,
        evidence_items: list[dict[str, str]],
        max_items: int = 4,
    ) -> list[dict[str, str]]:
        if not evidence_items:
            return []

        scored: list[tuple[int, dict[str, str]]] = []
        query_tokens = _tokenize(query)

        for item in evidence_items:
            overlap = len(query_tokens & _tokenize(item["title"] + " " + item["content"]))
            scored.append((overlap, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        selected = [item for score, item in scored[:max_items] if score > 0]

        if not selected:
            selected = [item for _, item in scored[:max_items]]

        return selected

    def _has_direct_support(self, answer: str, evidence_items: list[dict[str, str]]) -> bool:
        answer_text = _normalize_text(answer)
        answer_tokens = _extract_claim_tokens(answer)

        if not answer_tokens:
            return False

        for item in evidence_items:
            content = _normalize_text(item["content"])
            if answer_text and len(answer_text) > 12 and answer_text in content:
                return True

            overlap = sum(1 for token in answer_tokens if token in content)
            if overlap >= min(3, len(answer_tokens)):
                return True

        return False

    def _calibrate_critique(
        self,
        critique: dict[str, Any],
        answer: str,
        evidence_items: list[dict[str, str]],
    ) -> dict[str, Any]:
        calibrated = dict(critique)
        unsupported = calibrated.get("unsupported_claims", [])
        missing = calibrated.get("missing_evidence", [])
        parse_failure = any("could not be parsed" in item.lower() for item in unsupported if isinstance(item, str))
        direct_support = self._has_direct_support(answer, evidence_items)

        if direct_support and (parse_failure or (not unsupported and not missing)):
            calibrated["verdict"] = "VALID"
            calibrated["groundedness_score"] = max(int(calibrated.get("groundedness_score", 0)), 90)
            calibrated["unsupported_claims"] = [] if parse_failure else unsupported

        if unsupported and not parse_failure:
            calibrated["verdict"] = "PARTIALLY_SUPPORTED"
            calibrated["groundedness_score"] = min(int(calibrated.get("groundedness_score", 100)), 60)

        return calibrated

    def _format_evidence(self, evidence_items: list[dict[str, str]]) -> str:
        blocks = []
        for idx, item in enumerate(evidence_items, start=1):
            blocks.append(
                f"[Source {idx}] {item['title']} ({item['source_type']})\n"
                f"URL: {item['url']}\n"
                f"Evidence: {item['content']}"
            )
        return "\n\n".join(blocks)

    def _generate_grounded_answer(self, query: str, context: str) -> str:
        system_prompt = (
            "You are a careful grounded-answer generator. "
            "Answer the question clearly and concisely. "
            "Use only claims that are supported by the supplied evidence. "
            "If evidence is incomplete, explicitly qualify uncertainty instead of filling gaps with outside facts."
        )
        user_prompt = f"""Question:
{query}

Evidence:
{context}

Write a concise answer."""
        return self._chat(self.answer_model, system_prompt, user_prompt, max_tokens=500)

    def _critique_grounding(self, query: str, answer: str, context: str) -> dict[str, Any]:
        fallback = {
            "verdict": "PARTIALLY_SUPPORTED",
            "supported_claims": [],
            "unsupported_claims": ["Model critique could not be parsed as JSON."],
            "missing_evidence": [],
            "reasoning_issues": [],
            "groundedness_score": 50,
            "necessary_sources": [],
        }

        system_prompt = (
            "You are a strict grounding auditor. "
            "Check whether the answer is fully supported by the evidence only. "
            "Do not use outside knowledge. "
            "Return valid JSON only with keys: verdict, supported_claims, "
            "unsupported_claims, missing_evidence, reasoning_issues, groundedness_score, necessary_sources. "
            "Allowed verdict values: VALID, PARTIALLY_SUPPORTED, NOT_GROUNDED. "
            "groundedness_score must be an integer from 0 to 100. "
            "necessary_sources must be a list of source titles that were actually needed to support the answer. "
            "Treat both local KB evidence and external web evidence as valid support when relevant. "
            "Be conservative: if a claim is not directly supported, mark it unsupported."
        )
        user_prompt = f"""Question:
{query}

Evidence:
{context}

Answer to audit:
{answer}
"""
        return _safe_json_load(
            self._chat(self.check_model, system_prompt, user_prompt, max_tokens=700),
            fallback=fallback,
        )

    def _finalize_grounded_answer(
        self,
        query: str,
        draft_answer: str,
        critique: dict[str, Any],
        context: str,
    ) -> str:
        system_prompt = (
            "You are the final answer judge. "
            "Produce the best final answer using only supported claims. "
            "Remove unsupported claims instead of rephrasing them confidently. "
            "If evidence is partial, say so briefly and answer only the supported portion."
        )
        user_prompt = f"""Question:
{query}

Evidence:
{context}

Draft Answer:
{draft_answer}

Grounding Critique:
{json.dumps(critique, indent=2)}

Final grounded answer:"""
        return self._chat(self.answer_model, system_prompt, user_prompt, max_tokens=550)

    def _build_validity_summary(
        self,
        critique: dict[str, Any],
        final_answer: str,
        evidence_items: list[dict[str, str]],
    ) -> dict[str, Any]:
        score = critique.get("groundedness_score", 0)
        verdict = critique.get("verdict", "PARTIALLY_SUPPORTED")
        return {
            "verdict": verdict,
            "groundedness_score": score,
            "is_fully_grounded": verdict == "VALID" and score >= 85,
            "supported_claims": critique.get("supported_claims", []),
            "unsupported_claims": critique.get("unsupported_claims", []),
            "missing_evidence": critique.get("missing_evidence", []),
            "reasoning_issues": critique.get("reasoning_issues", []),
            "necessary_sources": critique.get("necessary_sources", []),
            "source_count": len(evidence_items),
            "external_source_count": sum(1 for item in evidence_items if item["source_type"] == "external_web"),
            "final_answer_preview": final_answer[:280],
        }

    def run(self, query: str, candidate_answer: str | None = None) -> dict[str, Any]:
        local_evidence = self.local_retriever.retrieve(query, top_k=self.retriever_top_k)
        external_evidence = self.external_retriever.retrieve(query)
        evidence_items = _dedupe_evidence(local_evidence + external_evidence)
        evidence_items = self._select_necessary_evidence(query, evidence_items)
        context = self._format_evidence(evidence_items)

        draft_answer = candidate_answer or self._generate_grounded_answer(query, context)
        critique = self._critique_grounding(query, draft_answer, context)
        critique = self._calibrate_critique(critique, draft_answer, evidence_items)
        final_answer = self._finalize_grounded_answer(query, draft_answer, critique, context)
        validity = self._build_validity_summary(critique, final_answer, evidence_items)

        return {
            "query": query,
            "retrieved_context": context,
            "retrieved_chunks": [
                f"{item['title']} ({item['source_type']}): {item['content']}" for item in evidence_items
            ],
            "evidence_items": evidence_items,
            "grounded_draft": draft_answer,
            "grounding_critique": critique,
            "final_answer": final_answer,
            "answer_validity": validity,
        }


def run_cli_demo() -> None:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in .env")

    pipeline = GroundedAnswerValidityPipeline(api_key=api_key)

    print("=" * 72)
    print("Grounded Answer Validity Demo")
    print("Type 'exit' to quit")
    print("=" * 72)

    while True:
        query = input("\nYou: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        result = pipeline.run(query)
        print("\nFinal Answer:\n")
        print(result["final_answer"])
        print("\nValidity:\n")
        print(json.dumps(result["answer_validity"], indent=2))


if __name__ == "__main__":
    run_cli_demo()
