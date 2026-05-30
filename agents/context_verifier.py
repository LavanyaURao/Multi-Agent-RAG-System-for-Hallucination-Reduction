"""
context_verifier.py
-------------------
Context Verifier Agent: filters retrieved evidence so the generator receives
only context that appears relevant to the current query.
"""

from groq import Groq


class ContextVerifierAgent:
    def __init__(self, client: Groq):
        self.client = client
        self.model = "llama-3.1-8b-instant"

    def verify(self, query: str, retrieved_items: list[dict]) -> dict:
        print("[ContextVerifierAgent] Verifying retrieved context...")

        if not retrieved_items:
            return {"verified_items": [], "summary": "No retrieved context available."}

        numbered_context = []
        for idx, item in enumerate(retrieved_items, start=1):
            numbered_context.append(
                f"[{idx}] source={item['source']} score={item['score']}\n{item['content']}"
            )

        system_prompt = (
            "You are a context verifier for a RAG system. "
            "Select only the context items that are relevant and non-redundant for answering the query. "
            "Return plain text with exactly two labels: KEEP: and SUMMARY:. "
            "KEEP must contain comma-separated item numbers. "
            "SUMMARY must briefly describe whether the remaining evidence is sufficient."
        )

        user_prompt = f"""Query:
{query}

Retrieved Context:
{chr(10).join(numbered_context)}

Decision:"""

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=220,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = (response.choices[0].message.content or "").strip()
        keep_indices, summary = self._parse(content, len(retrieved_items))
        verified_items = [retrieved_items[idx - 1] for idx in keep_indices if 1 <= idx <= len(retrieved_items)]

        if not verified_items:
            verified_items = retrieved_items[:2]
            if not summary:
                summary = "Verifier returned no usable selection, so the top retrieved items were retained."

        print(f"[ContextVerifierAgent] Retained {len(verified_items)} context items.")
        return {"verified_items": verified_items, "summary": summary}

    def _parse(self, content: str, total_items: int) -> tuple[list[int], str]:
        keep_indices: list[int] = []
        summary = ""

        for line in content.splitlines():
            if line.startswith("KEEP:"):
                raw = line.split(":", 1)[1].strip()
                for part in raw.split(","):
                    part = part.strip()
                    if part.isdigit():
                        keep_indices.append(int(part))
            elif line.startswith("SUMMARY:"):
                summary = line.split(":", 1)[1].strip()

        if not keep_indices and total_items > 0:
            keep_indices = [1]

        return keep_indices, summary
