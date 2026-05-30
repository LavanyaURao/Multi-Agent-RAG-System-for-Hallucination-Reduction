"""
query_understanding.py
----------------------
Query Understanding Agent: rewrites the user request into a structured form
that downstream retrieval and generation stages can use consistently.
"""

from groq import Groq


class QueryUnderstandingAgent:
    def __init__(self, client: Groq):
        self.client = client
        self.model = "llama-3.1-8b-instant"

    def understand(self, query: str) -> dict:
        print("[QueryUnderstandingAgent] Analyzing query...")

        system_prompt = (
            "You are a query understanding agent for a RAG pipeline. "
            "Extract the user's core intent and produce a retrieval-friendly rewrite. "
            "Classify whether the user input is primarily a question, factual lookup, explanatory request, comparative request, or procedural request. "
            "Also classify whether the answer is fact-sensitive. "
            "Return plain text using exactly these labels: "
            "INTENT:, QUERY_TYPE:, FACT_SENSITIVITY:, EXPECTED_ANSWER_STYLE:, REWRITTEN_QUERY:, KEY_TERMS:. "
            "KEY_TERMS must be a comma-separated list."
        )

        user_prompt = f"""User Query:
{query}

Structured Output:"""

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=180,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = (response.choices[0].message.content or "").strip()
        parsed = self._parse(content)
        parsed["original_query"] = query
        print("[QueryUnderstandingAgent] Query analysis complete.")
        return parsed

    def _parse(self, content: str) -> dict:
        result = {
            "intent": "General question answering",
            "query_type": "question",
            "fact_sensitivity": "medium",
            "expected_answer_style": "concise explanation",
            "rewritten_query": "",
            "key_terms": [],
        }

        for line in content.splitlines():
            if line.startswith("INTENT:"):
                result["intent"] = line.split(":", 1)[1].strip() or result["intent"]
            elif line.startswith("QUERY_TYPE:"):
                result["query_type"] = line.split(":", 1)[1].strip() or result["query_type"]
            elif line.startswith("FACT_SENSITIVITY:"):
                result["fact_sensitivity"] = line.split(":", 1)[1].strip() or result["fact_sensitivity"]
            elif line.startswith("EXPECTED_ANSWER_STYLE:"):
                result["expected_answer_style"] = line.split(":", 1)[1].strip() or result["expected_answer_style"]
            elif line.startswith("REWRITTEN_QUERY:"):
                result["rewritten_query"] = line.split(":", 1)[1].strip()
            elif line.startswith("KEY_TERMS:"):
                terms = line.split(":", 1)[1].strip()
                result["key_terms"] = [term.strip() for term in terms.split(",") if term.strip()]

        if not result["rewritten_query"]:
            result["rewritten_query"] = result["intent"]

        return result
