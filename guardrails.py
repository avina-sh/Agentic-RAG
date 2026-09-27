from pydantic import BaseModel,Field
from typing import Literal
from states import AgentState

from tools import llm
from config import MAX_QUESTION_LENGTH,PII_ENTITIES,SYSTEM_PROMPT_FINGERPRINTS,format_history
from prompts import INPUT_GUARDRAIL_PROMPT
from fallback import with_retry

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

analyzer=None
anonymizer=None


def get_pii_engines():
    global analyzer, anonymizer
    if analyzer is None:
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        anonymizer = AnonymizerEngine()

    return analyzer, anonymizer

def redact_pii(text:str)->tuple[str,bool]:
    analyzer,anonymizer=get_pii_engines()

    results=analyzer.analyze(
        text=text,
        entities=PII_ENTITIES,
        language="en"
    )

    if not results:
        return text,False

    redacted=anonymizer.anonymize(text,analyzer_results=results)

    return redacted.text,True

class SafetyCheck(BaseModel):
    is_unsafe:bool=Field(description="True if the input is unsafe")
    category:Literal["injection","jailbreak","leak","malicious_code","harmful_content","illegal_content","none"]

safety_llm=llm.with_structured_output(SafetyCheck,method="json_mode")

@with_retry(max_attempts=3, base_delay=1.0)
def safe_LLM_call(prompt):
    return safety_llm.invoke(prompt)


def check_input_safety(state:AgentState)->dict:
    question=state.question
    history_text=format_history(state.chat_history)
    if len(question) > MAX_QUESTION_LENGTH:
        return {"is_unsafe": True, "unsafe_category": "malicious_code"}

    try:
        result=safe_LLM_call(INPUT_GUARDRAIL_PROMPT.format(
            chat_history=history_text,
            question=question
        ))
        is_unsafe = result.is_unsafe
        category = result.category
    except Exception as e:
        print(f"[check_input_safety] Failed after retries: {e}")
        result = "I'm having trouble generating a response right now. Please try again in a moment."
        is_unsafe = True
        category = "none"

    print(f"[Guardrail : Input] is_unsafe={result.is_unsafe} category={result.category}")

    return {
        "is_unsafe":is_unsafe,
        "unsafe_category":category
    }

def decide_safety(state:AgentState)->str:
    return "unsafe" if state.is_unsafe else "safe"

def refuse(state:AgentState)->dict:
    category_messages = {
        "injection": "I can't follow instructions embedded in a question — I can only answer based on my knowledge base.",
        "jailbreak": "I'm not able to take on a different persona or bypass my guidelines.",
        "leak": "I can't share my internal instructions or system configuration.",
        "malicious_code": "I can't help with writing malicious code.",
        "harmful_content": "I can't help with that request.",
    }
    message=category_messages.get(state.unsafe_category,"I can't help with that request")
    return {"answer":message,"source_used":"refused"}

def check_output_safety(state:AgentState) -> dict:
    answer = state.answer.lower()
    leaked = any(fp in answer for fp in SYSTEM_PROMPT_FINGERPRINTS)

    if leaked:
        print("[Guardrail:Output] System prompt leak detected in output — replacing answer")
        return {"answer": "I can't share my internal instructions."}

    return {}