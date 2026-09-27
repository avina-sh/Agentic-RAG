from typing import List,Literal,Optional
from typing_extensions import TypedDict
from pydantic import BaseModel,Field
from langchain_core.documents import Document

class RouteDecision(BaseModel):
    route :Literal["kb","direct","web"]=Field(
        description="Use kb for questions needing Agentic RAG documents,direct for greetings/simple chat."
    )

class EvidenceGrade(BaseModel):
    grade:Literal["good","weak"]=Field(
        description="good means evidence can answer the question, weak means not enough evidence."
    )

class AgentState(BaseModel):
    question:str
    current_query:str
    kb_docs:List[Document]
    chat_history:List[dict]=[]
    web_results:str
    kb_grade:str
    web_grade:str
    answer:str
    is_unsafe:bool=False
    unsafe_category:str="None"
    source_used:str
    retry_count:int=0