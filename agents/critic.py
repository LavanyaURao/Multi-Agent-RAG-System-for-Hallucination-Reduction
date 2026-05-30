"""
critic.py
---------
Critic Agent: evaluates the generated answer for correctness,
hallucinations, and reasoning quality.
"""

from groq import Groq


class CriticAgent:
    def __init__(self, client: Groq):
        self.client = client
        self.model = "llama-3.1-8b-instant"

    def critique(self, query: str, answer: str) -> str:
        """
        Critique the generated answer using general knowledge.

        Args:
            query: Original user question.
            answer: Answer produced by the Generator.

        Returns:
            Critique listing issues found or confirmation of correctness.
        """
        print("[CriticAgent] Critiquing the generated answer...")

        system_prompt = (
            "You are a strict and intelligent critic. "
            "You will be given a question and an answer. "
            "Your job is to evaluate whether the answer is correct, logical, and free from hallucinations. "
            "Identify any incorrect facts, weak reasoning, or vague statements. "
            "List each issue on a separate line prefixed with '- ISSUE:'. "
            "If the answer is accurate and well-reasoned, respond with: "
            "'NO ISSUES FOUND. The answer is correct and well-reasoned.' "
            "Be concise and precise. Do NOT rewrite the answer."
        )

        user_prompt = f"""Question: {query}

Generated Answer:
{answer}

Critique:"""

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=512,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        critique = (response.choices[0].message.content or "").strip()
        print("[CriticAgent] Critique complete.")
        return critique

    def has_blocking_issues(self, critique: str) -> bool:
        normalized = critique.strip().upper()
        if normalized.startswith("NO ISSUES FOUND"):
            return False
        return "- ISSUE:" in normalized or "ISSUE" in normalized
