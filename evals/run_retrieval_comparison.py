import json
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import dense_search,hybrid_search,hybrid_search_with_rerank
from utils import load_golden_set,filter_by_type

def evaluate_retrieval_method(method_fn,cases,k=4,**kwargs):
    hits=0
    for case in cases:
        docs=method_fn(case["question"],k=k,**kwargs)
        retrieved_text=" ".join(doc.page_content.lower() for doc in docs)
        gt_keywords=case["ground_truth"].lower().split()[:8]
        overlap=sum(1 for word in gt_keywords if word in retrieved_text)
        hits += overlap/len(gt_keywords)
    return hits/len(cases)

def main():
    golden_set=load_golden_set()
    kb_cases=[c for c in filter_by_type(golden_set,"kb") if c.get("ground_truth")]

    methods={
        "dense":(dense_search,{}),
        "hybrid":(hybrid_search,{"alpha":0.5}),
        "hybrid_rerank":(hybrid_search_with_rerank,{"fetch_k":15,"alpha":0.5})
    }

    results={}
    for name, (fn,kwargs) in methods.items():
        score=evaluate_retrieval_method(fn,kb_cases,k=4,**kwargs)
        results[name]=round(score,3)
        print(f"{name}: {score:.3f}")

    with open("evals/results_retrieval_comparison.json","w") as f:
        json.dump(results,f,indent=2)

if __name__=="__main__":
    main()