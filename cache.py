import hashlib
import time
import json
import numpy as np
from database import get_embeddings

CACHE_STORE={}

SIMILARITY_THRESHOLD=0.92
TTL_BY_SOURCE={
    "kb":None,
    "direct":None,
    "refused":None,
    "web":60*60*3
}

def normalize(question:str)->str:
    return question.strip().lower()

def exact_cache_key(question:str)->str:
    return hashlib.sha256(normalize(question).encode()).hexdigest()

def cosine_similarity(a,b):
    a,b=np.array(a),np.array(b)
    return float(np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b)))

def is_expired(entry)->bool:
    ttl=TTL_BY_SOURCE.get(entry["source_used"])
    if ttl is None:
        return False
    return (time.time() - entry["timestamp"]) > ttl

def get_cached_answer(question:str):
    key=exact_cache_key(question)
    if key in CACHE_STORE and not is_expired(CACHE_STORE[key]):
        print(f"[CACHE] Exact hit: {question}")
        return CACHE_STORE[key]["result"]

    embeddings=get_embeddings()
    query_embedding=embeddings.embed_query(question)

    best_score=0
    best_entry=None
    for entry in CACHE_STORE.values():
        if is_expired(entry):
            continue
        score=cosine_similarity(query_embedding,entry["embedding"])
        if score>best_score:
            best_score=score
            best_entry=entry

    print(f"[Cache] Best semantic match score: {best_score:.4f} (threshold: {SIMILARITY_THRESHOLD}) — '{question}' vs '{best_entry['question'] if best_entry else 'N/A'}'")

    if best_entry and best_score > SIMILARITY_THRESHOLD:
        print(f"[CACHE] Semantic hit (score={best_score:.3f}): '{question}' ~ '{best_entry['question']}'")
        return best_entry["result"]

    return None

def store_in_cache(question:str,result:dict):
    embeddings=get_embeddings()
    key=exact_cache_key(question)
    CACHE_STORE[key]={
        "question":question,
        "embedding":embeddings.embed_query(question),
        "result":result,
        "timestamp":time.time(),
        "source_used":result.get("source_used","kb")
    }