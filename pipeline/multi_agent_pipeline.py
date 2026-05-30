"""
multi_agent_pipeline.py
-----------------------
Multi-Agent RAG Pipeline:
  1. QueryUnderstandingAgent -> structures the query
  2. RetrieverAgent          -> retrieves context with multiple retrievers
  3. ContextVerifierAgent    -> filters evidence before generation
  4. GeneratorAgent          -> produces the draft answer
  5. CriticAgent             -> checks correctness
  6. Feedback Loop           -> retries once if issues remain
  7. JudgeAgent              -> refines the final answer
"""

from groq import Groq

from agents.context_verifier import ContextVerifierAgent
from agents.critic import CriticAgent
from agents.generator import GeneratorAgent
from agents.judge import JudgeAgent
from agents.query_understanding import QueryUnderstandingAgent
from agents.retriever import RetrieverAgent
from database.vector_store import VectorStore


class MultiAgentPipeline:
    def __init__(self, api_key: str, knowledge_base_path: str = "data/knowledge_base.txt"):
        print("=" * 60)
        print("  Multi-Agent RAG Pipeline (Groq + Llama 3)")
        print("=" * 60)

        self.client = Groq(api_key=api_key)
        self.knowledge_base_path = knowledge_base_path

        self.vector_store = VectorStore()
        self.vector_store.load_and_index(self.knowledge_base_path)

        self.query_understanding = QueryUnderstandingAgent(self.client)
        self.retriever = RetrieverAgent(
            self.vector_store,
            knowledge_base_path=self.knowledge_base_path,
        )
        self.context_verifier = ContextVerifierAgent(self.client)
        self.generator = GeneratorAgent(self.client)
        self.critic = CriticAgent(self.client)
        self.judge = JudgeAgent(self.client)

        print("Pipeline ready.\n")

    def run(self, query: str) -> dict:
        print("=" * 60)
        print(f"Query: {query}")
        print("=" * 60)

        query_plan = self.query_understanding.understand(query)
        retrieved_items = self.retriever.retrieve(
            query=query_plan["rewritten_query"],
            key_terms=query_plan["key_terms"],
            top_k=3,
        )

        context_check = self.context_verifier.verify(query, retrieved_items)
        verified_items = context_check["verified_items"]
        verified_context = self._format_context(verified_items)

        initial_answer = self.generator.generate(
            query=query,
            context=verified_context,
            query_plan=query_plan,
        )

        critique = self.critic.critique(query, initial_answer)
        feedback_iterations = 0

        if self.critic.has_blocking_issues(critique):
            print("[FeedbackLoop] Critique found issues. Running one more retrieval/generation pass...")
            feedback_iterations = 1

            retry_query = f"{query_plan['rewritten_query']} {' '.join(query_plan['key_terms'])}".strip()
            retry_items = self.retriever.retrieve(
                query=retry_query,
                key_terms=query_plan["key_terms"],
                top_k=4,
            )
            retry_check = self.context_verifier.verify(query, retry_items)
            verified_items = retry_check["verified_items"]
            verified_context = self._format_context(verified_items)

            initial_answer = self.generator.generate(
                query=query,
                context=verified_context,
                query_plan=query_plan,
            )
            critique = self.critic.critique(query, initial_answer)
            context_check = retry_check
            retrieved_items = retry_items

        final_answer = self.judge.judge(query, initial_answer, critique)

        result = {
            "query": query,
            "query_plan": query_plan,
            "retrieved_context_items": retrieved_items,
            "verified_context_items": verified_items,
            "context_verifier_summary": context_check["summary"],
            "initial_answer": initial_answer,
            "critique": critique,
            "final_answer": final_answer,
            "feedback_iterations": feedback_iterations,
        }

        self._print_result(result)
        return result

    def _format_context(self, context_items: list[dict]) -> str:
        if not context_items:
            return "No verified context available."

        blocks = []
        for idx, item in enumerate(context_items, start=1):
            blocks.append(
                f"[Context {idx}] Source: {item['source']} | Score: {item['score']}\n"
                f"{item['content']}"
            )
        return "\n\n".join(blocks)

    def _print_result(self, result: dict):
        print("\n" + "=" * 60)
        print("PIPELINE RESULTS")
        print("=" * 60)
        print(f"\n[QUERY]\n{result['query']}\n")
        print(f"[QUERY PLAN]\n{result['query_plan']}\n")
        print(f"[CONTEXT VERIFIER]\n{result['context_verifier_summary']}\n")
        print(f"[INITIAL ANSWER]\n{result['initial_answer']}\n")
        print(f"[CRITIC FEEDBACK]\n{result['critique']}\n")
        print(f"[FINAL ANSWER]\n{result['final_answer']}\n")
        print("=" * 60)
