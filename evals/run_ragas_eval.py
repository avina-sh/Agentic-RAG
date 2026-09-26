import json
from openai import AsyncOpenAI

from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils import load_golden_set, run_agent_timed, filter_by_type


def build_ragas_dataset(cases):
    samples = []
    for case in cases:
        if not case.get("ground_truth"):
            continue
        result, _ = run_agent_timed(case["question"])
        samples.append(
            SingleTurnSample(
                user_input=case["question"],
                response=result["answer"],
                retrieved_contexts=[doc.page_content for doc in result.get("kb_docs", [])],
                reference=case["ground_truth"],
            )
        )
    return EvaluationDataset(samples=samples)


def main():
    golden_set = load_golden_set()
    kb_cases = filter_by_type(golden_set, "kb") + filter_by_type(golden_set, "false_positive_check")

    print(f"Evaluating {len(kb_cases)} KB-answerable questions with RAGAS...")
    dataset = build_ragas_dataset(kb_cases)

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]

    results = evaluate(dataset=dataset, metrics=metrics)

    scores = results.to_pandas().mean(numeric_only=True).to_dict()
    print(json.dumps(scores, indent=2))

    with open("evals/results_ragas.json", "w") as f:
        json.dump(scores, f, indent=2)


if __name__ == "__main__":
    main()