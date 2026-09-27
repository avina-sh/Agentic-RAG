import streamlit as st
from langsmith import Client
from langchain_core.tracers.context import collect_runs

from database import load_retriever, get_reranker, load_vector_store
from workflow import ask_agent

st.set_page_config(page_title="Agentic RAG", page_icon="🔎", layout="centered")

st.title("Agentic RAG Assistant")
st.caption("Ask questions grounded in the knowledge base, with automatic web fallback and citation checking.")

langsmith_client = Client()

SOURCE_BADGES = {
    "kb": ("📚 Knowledge Base", "#2563eb"),
    "web": ("🌐 Web Search", "#059669"),
    "direct": ("💬 Direct", "#6b7280"),
    "refused": ("🛡️ Blocked by Guardrail", "#dc2626"),
    "error": ("⚠️ Error", "#dc2626"),
}

EXAMPLE_QUESTIONS = [
    "What is agentic RAG?",
    "How does query rewriting work?",
    "What's the latest AI news?",
]


@st.cache_resource
def get_retriever():
    return load_retriever()


@st.cache_resource(show_spinner=False)
def warm_up_models():
    reranker = get_reranker()
    vectorstore = load_vector_store()

    vectorstore.similarity_search("warmup", k=1)
    reranker.predict([["warmup_query", "warmup document text"]])

    return True


def ask_agent_with_trace(question: str):
    with collect_runs() as run_collector:
        result = ask_agent(question)

    trace_url = None
    if run_collector.traced_runs:
        root_run = run_collector.traced_runs[0]
        try:
            trace_url = langsmith_client.get_run_url(run=root_run)
        except Exception as e:
            print(f"[LangSmith] Could not get trace URL: {e}")

    return result, trace_url


def render_source_badge(source_used: str):
    label, color = SOURCE_BADGES.get(source_used, ("Unknown", "#6b7280"))
    st.markdown(
        f'<span style="background-color:{color}20; color:{color}; padding:3px 12px; '
        f'border-radius:12px; font-size:0.8em; font-weight:600;">{label}</span>',
        unsafe_allow_html=True,
    )


def render_sources(result: dict):
    docs = result.get("kb_docs") or []
    if not docs:
        return
    with st.expander("📄 Sources"):
        seen = set()
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            if source not in seen:
                st.markdown(f"- {source}")
                seen.add(source)


def render_message(msg: dict):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("source_used"):
            render_source_badge(msg["source_used"])
        if msg.get("kb_docs"):
            render_sources({"kb_docs": msg["kb_docs"]})
        if st.session_state.get("debug_mode") and msg.get("trace_url"):
            st.markdown(f"[🔍 View reasoning trace]({msg['trace_url']})")

with st.sidebar:
    st.markdown("### About this project")
    st.markdown(
        "Agentic RAG assistant with hybrid search + reranking, "
        "input/output guardrails, PII redaction, semantic caching, "
        "and automatic web fallback for time-sensitive questions.\n\n"
    )
    st.checkbox("🔧 Debug mode (show trace links)", key="debug_mode")

with st.spinner("Connecting to knowledge base..."):
    retriever = get_retriever()
    warm_up_models()

st.success("Retriever ready ✅")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, example in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(example, use_container_width=True):
            st.session_state.pending_question = example
            st.rerun()

for msg in st.session_state.messages:
    render_message(msg)

question = st.chat_input("Ask a question...")
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result, trace_url = ask_agent_with_trace(question)
                answer = result["answer"]
                source_used = result.get("source_used")
                kb_docs = result.get("kb_docs")

                st.markdown(answer)
                if source_used:
                    render_source_badge(source_used)
                if kb_docs:
                    render_sources(result)
                if st.session_state.get("debug_mode") and trace_url:
                    st.markdown(f"[🔍 View reasoning trace]({trace_url})")

            except Exception as e:
                answer = "Something went wrong processing your question. Please try again in a moment."
                source_used = "error"
                kb_docs = None
                trace_url = None
                print(f"[app.py] Unhandled error: {e}")
                st.markdown(answer)
                render_source_badge(source_used)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "source_used": source_used,
            "kb_docs": kb_docs,
            "trace_url": trace_url,
        }
    )