"""
generator.py
------------
Generator Agent: uses Groq (Llama 3) to produce an answer grounded in the
verified context passed by the pipeline.
"""

from groq import Groq


class GeneratorAgent:
    def __init__(self, client: Groq):
        self.client = client
        self.model = "llama-3.3-70b-versatile"

    def generate(self, query: str, context: str, query_plan: dict | None = None) -> str:
        """
        Generate an answer grounded in verified context.

        Args:
            query: The user's question.
            context: Verified evidence selected by retrieval.
            query_plan: Structured query understanding output.

        Returns:
            Generated answer as a string.
        """
        print("[GeneratorAgent] Generating answer using verified context...")

        plan_text = ""
        if query_plan:
            plan_text = (
                f"Intent: {query_plan.get('intent', '')}\n"
                f"Rewritten Query: {query_plan.get('rewritten_query', '')}\n"
                f"Key Terms: {', '.join(query_plan.get('key_terms', []))}\n\n"
            )

        system_prompt = (
            "You are a helpful, accurate RAG generator. "
            "Answer the user's question clearly and concisely using the provided context as primary evidence. "
            "If the context is incomplete, you may add limited general knowledge only when it does not conflict with the evidence. "
            "Do not invent citations or unsupported facts."
        )

        user_prompt = f"""{plan_text}Question: {query}

Verified Context:
{context}

Answer:"""

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=512,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = (response.choices[0].message.content or "").strip()
        print("[GeneratorAgent] Answer generated.")
        return answer
