import os
from dotenv import load_dotenv
from typing import Literal
from states import RouteDecision,EvidenceGrade,AgentState
from langgraph.graph import StateGraph,START,END
from tools import llm,web_search
from prompts import ROUTER_PROMPT_TEMPLATE,GRADER_PROMPT_TEMPLATE,REWRITE_PROMPT_TEMPLATE,PRIVATE_KB_PROMPT_TEMPLATE,WEB_SEARCH_PROMPT_TEMPLATE
from database import load_retriever
from config import MAX_RETRIES
from guardrails import check_input_safety, decide_safety, refuse, check_output_safety,redact_pii
from cache import get_cached_answer, store_in_cache
from fallback import with_retry

load_dotenv(override=True)

router_llm=llm.with_structured_output(RouteDecision,method="json_mode")
grader_llm=llm.with_structured_output(EvidenceGrade,method="json_mode")


llm_retry = with_retry(max_attempts=3, base_delay=1.0)
search_retry = with_retry(max_attempts=2, base_delay=1.5)

@llm_retry
def safe_router_call(prompt):
    return router_llm.invoke(prompt)

@llm_retry
def safe_grader_call(prompt):
    return grader_llm.invoke(prompt)

@llm_retry
def safe_LLM_call(prompt):
    return llm.invoke(prompt)

@search_retry
def safe_tavily_call(query):
    return web_search.invoke(query)


def check_pii(state:AgentState)->dict:
    redacted_question,found=redact_pii(state.question)
    if found:
        print(f"[Guardrail : PII] Redacted PII from input")

    return {"current_query":redacted_question}

def route_question(state:AgentState):
    question=state.current_query

    ROUTER_PROMPT=ROUTER_PROMPT_TEMPLATE.format(question=question)

    try:
        decision=safe_router_call(ROUTER_PROMPT)
        route=decision.route
    except Exception as e:
        print(f"[route_question] Failed after retries {e}")
        route="kb"

    print("[Router]",decision.route)

    return {
        "current_query":question,
        "source_used":route
    }


def route_after_router(state:AgentState)->Literal["retrieve_kb","search_web","direct_answer"]:
    if state.source_used=="kb":
        return "retrieve_kb"
    elif state.source_used=="web":
        return "search_web"
    return "direct_answer"

def retrieve_kb(state:AgentState):
    query=state.current_query
    retriever=load_retriever()
    docs=retriever.invoke(query)

    print(f"[KB Retriever] Query: {query}")
    print(f"[KB Retriever] Retrieved: {len(docs)} chunks")

    return {"kb_docs": docs}


def grade_kb_evidence(state:AgentState):
    question=state.current_query

    context="\n\n".join(
        f"Source: {doc.metadata.get('source')}\n{doc.page_content}"
        for doc in state.kb_docs
    )

    GRADER_PROMPT=GRADER_PROMPT_TEMPLATE.format(question=question,context=context)

    try:
        grade=safe_grader_call(GRADER_PROMPT)
    except Exception as e:
        print(f"[grade_kb_evidence] Failed after retries: {e}")
        grade = "weak" 

    print("[KB Grader]", grade.grade)
    return {"kb_grade": grade.grade}


def decide_after_kb_grade(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    if state.kb_grade == "good":
        return "generate_from_kb"
    return "search_web"


def search_web(state:AgentState):
    query=state.current_query

    try:
        result=safe_tavily_call(query)

        if isinstance(result,dict):
            answer=result.get("answer","")
            results=result.get("results",[])
            lines=[]
            if answer:
                lines.append(f"Tavily answer: {answer}")

            for item in results:
                title=item.get("title","")
                url=item.get("url","")
                content=item.get("content","")
                lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")

            web_text="\n\n".join(lines) if lines else str(result)
        else:
            web_text=str(result)
    except Exception as e:
        print(f"[search_web] Failed after retries: {e}")
        return {
            "web_results": "",
            "answer": "I couldn't find current information for this — both the knowledge base and web search are unavailable right now.",
            "source_used": "error",
        }

    print("[Tavily Search] Result characters:", len(web_text))

    return {
        "web_results": web_text,
        "source_used": "web",
    }


def grade_web_evidence(state: AgentState):
    question=state.current_query
    web_results = state.web_results

    GRADER_PROMPT=GRADER_PROMPT_TEMPLATE.format(question=question,context=web_results)

    try:
        grade=safe_grader_call(GRADER_PROMPT)
    except Exception as e:
        print(f"[grade_web_evidence] Failed after retries: {e}")
        grade = "weak"
    
    print("[Web Grader]", grade.grade)

    return {"web_grade": grade.grade}


def decide_after_web_grade(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "answer_insufficient"]:
    if state.web_grade == "good":
        return "generate_from_web"

    if state.retry_count < MAX_RETRIES:
        return "rewrite_query"

    return "answer_insufficient"


def rewrite_query(state:AgentState):
    question=state.current_query
    retry_count=state.retry_count + 1

    REWRITE_PROMPT=REWRITE_PROMPT_TEMPLATE.format(question=question)

    try:
        rewritten_query=safe_LLM_call(REWRITE_PROMPT).content.strip()
    except Exception as e:
        print(f"[rewrite_query] Failed after retries: {e}")
        rewritten_query = "I'm having trouble generating a response right now. Please try again in a moment."    

    print("[Rewriter]", rewritten_query)

    return {
        "current_query": rewritten_query,
        "retry_count": retry_count,
    }


def generate_from_kb(state:AgentState):
    question=state.current_query
    context="\n\n".join(
        f"[KB Source :{doc.metadata.get('source')}]\n{doc.page_content}"
        for doc in state.kb_docs
    )
    PRIVATE_KB_PROMPT=PRIVATE_KB_PROMPT_TEMPLATE.format(
        question=question,
        context=context
    )

    try:
        answer=safe_LLM_call(PRIVATE_KB_PROMPT).content
    except Exception as e:
        print(f"[generate_from_kb] Failed after retries: {e}")
        answer = "I'm having trouble generating a response right now. Please try again in a moment."

    return {
        "answer":answer,
        "source_used":"private_kb"
    }


def generate_from_web(state: AgentState):
    question=state.current_query
    web_context = state.web_results

    WEB_SEARCH_PROMPT=WEB_SEARCH_PROMPT_TEMPLATE.format(
        question=question,
        web_context=web_context
    )

    try:
        answer=safe_LLM_call(WEB_SEARCH_PROMPT).content
    except Exception as e:
        print(f"[generate_from_web] Failed after retries: {e}")
        answer = "I'm having trouble generating a response right now. Please try again in a moment."

    return {
        "answer": answer,
        "source_used": "web_search",
    }

def direct_answer(state: AgentState):
    question=state.current_query

    try:
        answer = safe_LLM_call(
            f'''
            Respond briefly and naturally.
            Message:{question}'''
        ).content
    except Exception as e:
        print(f"[direct_answer] Failed after retries: {e}")
        answer = "I'm having trouble generating a response right now. Please try again in a moment."

    return {
        "answer": answer,
        "source_used": "direct",
    }

def answer_insufficient(state: AgentState):
    answer = (
        "I could not find enough reliable evidence in the private knowledge base "
        "or the web search results to answer this confidently. "
        "Please provide more specific documents or rephrase the question."
    )

    return {
        "answer": answer,
        "source_used": "insufficient_evidence",
    }

workflow=StateGraph(AgentState)

workflow.add_node("check_pii",check_pii)
workflow.add_node("check_input_safety", check_input_safety)
workflow.add_node("refuse", refuse)
workflow.add_node("check_output_safety", check_output_safety)
workflow.add_node("route_question", route_question)
workflow.add_node("retrieve_kb", retrieve_kb)
workflow.add_node("grade_kb_evidence", grade_kb_evidence)
workflow.add_node("search_web", search_web)
workflow.add_node("grade_web_evidence", grade_web_evidence)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("generate_from_kb", generate_from_kb)
workflow.add_node("generate_from_web", generate_from_web)
workflow.add_node("direct_answer", direct_answer)
workflow.add_node("answer_insufficient", answer_insufficient)

workflow.add_edge(START, "check_input_safety")

workflow.add_conditional_edges(
    "check_input_safety",
    decide_safety,
    {
        "unsafe": "refuse",
        "safe": "check_pii",   
    }
)

workflow.add_edge("check_pii","route_question")

workflow.add_conditional_edges(
    "route_question",
    route_after_router,
    {
        "retrieve_kb":"retrieve_kb",
        "search_web":"search_web",
        "direct_answer":"direct_answer",
    },
)

workflow.add_edge("retrieve_kb", "grade_kb_evidence")

workflow.add_conditional_edges(
    "grade_kb_evidence",
    decide_after_kb_grade,
    {
        "generate_from_kb": "generate_from_kb",
        "search_web": "search_web",
    },
)

workflow.add_edge("search_web", "grade_web_evidence")

workflow.add_conditional_edges(
    "grade_web_evidence",
    decide_after_web_grade,
    {
        "generate_from_web": "generate_from_web",
        "rewrite_query": "rewrite_query",
        "answer_insufficient": "answer_insufficient",
    },
)

workflow.add_edge("rewrite_query", "retrieve_kb")

workflow.add_edge("generate_from_kb", "check_output_safety")
workflow.add_edge("generate_from_web", "check_output_safety")
workflow.add_edge("direct_answer", "check_output_safety")

workflow.add_edge("answer_insufficient", END)
workflow.add_edge("refuse",END)
workflow.add_edge("check_output_safety", END)

graph = workflow.compile()

print("Industry-style Agentic RAG graph compiled.")

def ask_agent(question:str):

    cached=get_cached_answer(question)
    if cached:
        return cached
    
    initial_state:AgentState={
        "question":question,
        "current_query":question,
        "kb_docs":[],
        "web_results":"",
        "kb_grade":"",
        "web_grade":"",
        "answer":"",
        "source_used":"",
        "retry_count":0
    }

    result = graph.invoke(initial_state)

    store_in_cache(question,result)

    print("\n" + "=" * 90)
    print("QUESTION:")
    print(question)
    print("\nSOURCE USED:")
    print(result["source_used"])
    print("\nFINAL ANSWER:")
    print(result["answer"])
    print("=" * 90)

    return result

def view_graph():
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_bytes)

if __name__=="__main__":
    view_graph()