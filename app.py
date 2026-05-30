import os

import streamlit as st
from dotenv import load_dotenv

from grounded_answer_validity_demo import GroundedAnswerValidityPipeline
from pipeline.multi_agent_pipeline import MultiAgentPipeline


load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    st.error("Please set your GROQ_API_KEY in .env file")
    st.stop()


@st.cache_resource
def load_pipeline():
    return MultiAgentPipeline(api_key=API_KEY)


def load_eagv_pipeline():
    return GroundedAnswerValidityPipeline(api_key=API_KEY)


pipeline = load_pipeline()
eagv_pipeline = load_eagv_pipeline()

st.title("Multi-Agent LLM Chat")
st.write("Groq + Llama 3 Multi-Agent RAG Pipeline with EAGV")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def _render_source_list(items):
    for idx, item in enumerate(items, start=1):
        st.markdown(f"**Source {idx}: {item['title']}**")
        st.write(f"Type: {item['source_type']}")
        st.write(item["url"])
        st.write(item["content"])

query = st.chat_input("Type your question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Thinking..."):
        result = pipeline.run(query)
        eagv_result = eagv_pipeline.run(query, candidate_answer=result["final_answer"])

    final_answer = result["final_answer"]

    with st.chat_message("assistant"):
        st.markdown(final_answer)

        with st.expander("Query Understanding", expanded=True):
            plan = result["query_plan"]
            col1, col2 = st.columns(2)
            col1.metric("Query Type", plan.get("query_type", "unknown"))
            col2.metric("Fact Sensitivity", plan.get("fact_sensitivity", "unknown"))
            st.write(f"Intent: {plan.get('intent', '')}")
            st.write(f"Expected Answer Style: {plan.get('expected_answer_style', '')}")
            st.write(f"Rewritten Query: {plan.get('rewritten_query', '')}")
            st.write("Key Terms:")
            st.write(", ".join(plan.get("key_terms", [])) or "None")

        with st.expander("Multi-Retriever Output"):
            unique_items = []
            seen = set()
            for item in result["retrieved_context_items"]:
                key = (item["source"], item["content"].strip().lower())
                if key in seen:
                    continue
                seen.add(key)
                unique_items.append(item)

            for idx, item in enumerate(unique_items, start=1):
                st.markdown(f"**Item {idx}**")
                st.write(f"Source: {item['source']}")
                st.write(f"Score: {item['score']}")
                st.write(item["content"])

        with st.expander("Context Verifier"):
            st.write(result["context_verifier_summary"])
            for idx, item in enumerate(result["verified_context_items"], start=1):
                st.markdown(f"**Verified Context {idx}**")
                st.write(f"Source: {item['source']}")
                st.write(f"Score: {item['score']}")
                st.write(item["content"])

        with st.expander("Initial Answer"):
            st.write(result["initial_answer"])

        with st.expander("Critique"):
            st.write(result["critique"])

        with st.expander("Feedback Loop"):
            st.write(
                "Feedback loop triggered."
                if result["feedback_iterations"] > 0
                else "Feedback loop not triggered."
            )
            st.write(f"Retry count: {result['feedback_iterations']}")

        with st.expander("EAGV Verdict", expanded=True):
            validity = eagv_result["answer_validity"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Verdict", validity["verdict"])
            col2.metric("Groundedness", f'{validity["groundedness_score"]}/100')
            col3.metric("Fully Grounded", "Yes" if validity["is_fully_grounded"] else "No")
            st.write("Necessary Sources:")
            st.write(", ".join(validity.get("necessary_sources", [])) or "None identified")

        with st.expander("EAGV Retrieved Evidence"):
            for chunk in eagv_result["retrieved_chunks"]:
                st.write(chunk)

        with st.expander("EAGV Sources Used"):
            necessary_titles = set(validity.get("necessary_sources", []))
            filtered_items = [
                item for item in eagv_result["evidence_items"]
                if not necessary_titles or item["title"] in necessary_titles
            ]
            _render_source_list(filtered_items)

        with st.expander("EAGV Grounding Critique"):
            st.json(eagv_result["grounding_critique"])

        with st.expander("EAGV Grounded Rewrite"):
            st.write(eagv_result["final_answer"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer
    })
