"""
judge.py
--------
Judge Agent: Produces the final refined answer by combining
the initial answer and critic feedback using LLM reasoning.

(NO RAG / NO CONTEXT)
"""

from groq import Groq


class JudgeAgent:
    def __init__(self, client: Groq):
        self.client = client
        self.model = "llama-3.1-8b-instant"

    def judge(self, query: str, answer: str, critique: str) -> str:
        """
        Produce a final answer by refining the initial answer
        based on critic feedback.

        Args:
            query: Original user question.
            answer: Initial answer from Generator.
            critique: Feedback from Critic.

        Returns:
            Final refined answer as a string.
        """
        print("[JudgeAgent] Producing final judgment...")

        system_prompt = (
            "You are an intelligent and impartial judge. "
            "You are given a question, an initial answer, and a critic's feedback. "
            "Your task is to produce the best possible final answer. "
            "If the critic identifies issues, fix them in the final answer. "
            "If there are no issues, return a clean and improved version of the original answer. "
            "Ensure the final answer is accurate, clear, and well-structured. "
            "Do NOT mention the critic or the process in your final response."
        )

        user_prompt = f"""Question: {query}

Initial Answer:
{answer}

Critic Feedback:
{critique}

Final Answer:"""

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=600,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        final_answer = response.choices[0].message.content.strip()
        print("[JudgeAgent] Final answer produced.")
        return final_answer