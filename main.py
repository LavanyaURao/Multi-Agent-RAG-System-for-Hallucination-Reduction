"""
Interactive Multi-Agent LLM system with EAGV output.

Type 'exit' to quit.
"""

import json
import os

from dotenv import load_dotenv

from grounded_answer_validity_demo import GroundedAnswerValidityPipeline
from pipeline.multi_agent_pipeline import MultiAgentPipeline


load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")

if API_KEY == "your_groq_api_key_here":
    print("ERROR: Please set your GROQ_API_KEY in the .env file.")
    print("Get a free key at https://console.groq.com")
    raise SystemExit(1)


def main():
    print("=" * 60)
    print("Multi-Agent LLM Chat (Groq + Llama 3 + EAGV)")
    print("Type 'exit' to quit")
    print("=" * 60)

    pipeline = MultiAgentPipeline(api_key=API_KEY)
    eagv_pipeline = GroundedAnswerValidityPipeline(api_key=API_KEY)

    all_results = []

    while True:
        query = input("\nYou: ")

        if query.lower() in ["exit", "quit"]:
            print("\nExiting... Bye!")
            break

        if not query.strip():
            print("Please enter a valid question.")
            continue

        result = pipeline.run(query)
        eagv_result = eagv_pipeline.run(query, candidate_answer=result["final_answer"])

        formatted_result = {
            "query": result["query"],
            "query_plan": result["query_plan"],
            "retrieved_context_items": result["retrieved_context_items"],
            "verified_context_items": result["verified_context_items"],
            "context_verifier_summary": result["context_verifier_summary"],
            "initial_answer": result["initial_answer"],
            "critique": result["critique"],
            "final_answer": result["final_answer"],
            "feedback_iterations": result["feedback_iterations"],
            "eagv_grounded_rewrite": eagv_result["final_answer"],
            "answer_validity": eagv_result["answer_validity"],
            "retrieved_chunks": eagv_result["retrieved_chunks"],
            "evidence_items": eagv_result["evidence_items"],
            "grounding_critique": eagv_result["grounding_critique"],
        }

        all_results.append(formatted_result)

        print("\nFinal Answer:")
        print(formatted_result["final_answer"])
        print("\nEAGV:")
        print(json.dumps(formatted_result["answer_validity"], indent=2))

    output_path = "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nAll chat results saved to {output_path}")


if __name__ == "__main__":
    main()
