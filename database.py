import os
from dotenv import load_dotenv
import hashlib
import time
import pickle
from pydantic import ConfigDict
from sentence_transformers import CrossEncoder

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from config import INDEX_NAME, NAMESPACE,SOURCE_URLS,EMBEDDING_DIM,BM25_CACHE_PATH
from rank_bm25 import BM25Okapi

load_dotenv(override=True)
os.environ.setdefault("USER_AGENT", "Mozilla/5.0 Agentic-RAG-Demo")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

reranker=None
vectorstore=None

class HybridRerankRetriever(BaseRetriever):
    k:int=4
    fetch_k:int=15
    alpha:float=0.5

    model_config=ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self,query:str,*,run_manager=None):
        del run_manager
        return hybrid_search_with_rerank(
            query,
            k=self.k,
            fetch_k=self.fetch_k,
            alpha=self.alpha
        )

    
def get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )


def ingest():
    raw_docs=[]
    for url in SOURCE_URLS:
        loader = WebBaseLoader(
            web_paths=(url,),
            requests_kwargs={
                "headers": {
                    "User-Agent": "Mozilla/5.0 Agentic RAG Demo"
                }
            }
        )
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"]=url

        raw_docs.extend(docs)

    print(f"Loaded {len(raw_docs)} documents from {len(SOURCE_URLS)} URLs")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True
    )
    chunks = splitter.split_documents(raw_docs)

    build_and_cache_bm25(chunks)

    embeddings = get_embeddings()

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            )
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)

    ids = [
        hashlib.sha256(
            f"{doc.metadata.get('source')}::{doc.metadata.get('start_index')}".encode()
        ).hexdigest()
        for doc in chunks
    ]

    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=NAMESPACE,
        ids=ids
    )

    print(f"Ingested {len(chunks)} chunks into '{INDEX_NAME}' / namespace '{NAMESPACE}'")
    return vectorstore


def load_vector_store():
    global vectorstore
    if vectorstore is None:
        embeddings=get_embeddings()

        pc=Pinecone(api_key=PINECONE_API_KEY)
        index=pc.Index(INDEX_NAME)

        vectorstore=PineconeVectorStore(
            index=index,
            embedding=embeddings,
            namespace=NAMESPACE
        )
    return vectorstore

def dense_search(query,k=4):
    vectorstore=load_vector_store()

    docs=vectorstore.similarity_search(
        query,
        k=k
    )
    return docs


def load_retriever():
    return HybridRerankRetriever(k=4,fetch_k=15,alpha=0.5)


def build_and_cache_bm25(chunks):
    tokenized_chunks=[doc.page_content.lower().split() for doc in chunks]
    bm25=BM25Okapi(tokenized_chunks)

    with open(BM25_CACHE_PATH,"wb") as f:
        pickle.dump({"bm25":bm25,"chunks":chunks},f)

    print(f"BM25 index cached to {BM25_CACHE_PATH}")
    return bm25,chunks

def load_bm25():
    if not os.path.exists(BM25_CACHE_PATH):
        print("[BM25] Cache not found — building from source URLs...")
        ingest()

        if not os.path.exists(BM25_CACHE_PATH):
            raise RuntimeError(
                f"BM25 cache still missing at {BM25_CACHE_PATH} after ingest() — "
                f"check that build_and_cache_bm25() is being called and writing to the correct path."
            )

    with open(BM25_CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    return cache["bm25"], cache["chunks"]


def sparse_search(query,k=4):
    bm25,chunks=load_bm25()

    query_tokens=query.lower().split()

    scores=bm25.get_scores(query_tokens)

    top_indices=sorted(
        range(len(scores)),
        key=lambda i : scores[i],
        reverse=True
    )[:k]

    return [chunks[i] for i in top_indices]

def hybrid_search(query, k=4, alpha=0.5):
    dense_docs = dense_search(query, k=10)
    sparse_docs = sparse_search(query, k=10)

    dense_map = {
        (
            doc.metadata.get("source"),
            doc.metadata.get("start_index")
        ): doc
        for doc in dense_docs
    }

    sparse_map = {
        (
            doc.metadata.get("source"),
            doc.metadata.get("start_index")
        ): doc
        for doc in sparse_docs
    }

    bm25, chunks = load_bm25()

    query_tokens = query.lower().split()
    sparse_scores = bm25.get_scores(query_tokens)

    sparse_score_map = {
        (
            doc.metadata.get("source"),
            doc.metadata.get("start_index")
        ): score
        for doc, score in zip(chunks, sparse_scores)
    }

    all_ids = set(dense_map.keys()) | set(sparse_map.keys())

    results = []

    for chunk_id in all_ids:

        doc = dense_map.get(chunk_id) or sparse_map.get(chunk_id)

        dense_rank = (
            list(dense_map.keys()).index(chunk_id) + 1
            if chunk_id in dense_map
            else 100
        )

        sparse_rank = (
            list(sparse_map.keys()).index(chunk_id) + 1
            if chunk_id in sparse_map
            else 100
        )

        dense_score = 1 / dense_rank
        sparse_score = 1 / sparse_rank

        hybrid_score = (
            alpha * dense_score +
            (1 - alpha) * sparse_score
        )

        results.append((hybrid_score, doc))

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for score, doc in results[:k]]


def get_reranker():
    global reranker
    if reranker is None:
        reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return reranker

def hybrid_search_with_rerank(query,k=4,fetch_k=10,alpha=0.5):
    # t0 = time.perf_counter()
    dense_docs = dense_search(query, k=fetch_k)
    # t1 = time.perf_counter()
    sparse_docs = sparse_search(query, k=fetch_k)
    # t2 = time.perf_counter()


    dense_map={
        (doc.metadata.get("source"),doc.metadata.get("start_index")): doc
        for doc in dense_docs
    }

    sparse_map={
        (doc.metadata.get("source"),doc.metadata.get("start_index")): doc
        for doc in sparse_docs
    }

    dense_ids=list(dense_map.keys())
    sparse_ids=list(sparse_map.keys())
    all_ids=set(dense_ids) | set(sparse_ids)

    fused=[]
    for chunk_id in all_ids:
        doc = dense_map.get(chunk_id) or sparse_map.get(chunk_id)
        dense_rank = dense_ids.index(chunk_id) + 1 if chunk_id in dense_map else fetch_k + 1
        sparse_rank = sparse_ids.index(chunk_id) + 1 if chunk_id in sparse_map else fetch_k + 1
        hybrid_score = alpha * (1 / dense_rank) + (1 - alpha) * (1 / sparse_rank)
        fused.append((hybrid_score,doc))

    fused.sort(key=lambda x : x[0],reverse=True)

    fused_docs=[doc for _,doc in fused[:fetch_k]]

    reranker=get_reranker()

    # t3 = time.perf_counter()

    scores=reranker.predict([[query,doc.page_content] for doc in fused_docs])

    # t4 = time.perf_counter()

    # print(f"[Timing] dense={t1-t0:.3f}s sparse={t2-t1:.3f}s fusion={t3-t2:.3f}s rerank={t4-t3:.3f}s")

    reranked=sorted(zip(scores,fused_docs),key=lambda x:float(x[0]),reverse=True)

    final_docs=[]
    for score,doc in reranked[:k]:
        doc.metadata["reranker_score"]=float(score)
        final_docs.append(doc)

    return final_docs

if __name__ == "__main__":
    ingest()
    # query = "Explain semantic search in RAG applications"
    # docs = hybrid_search_with_rerank(query, k=4, fetch_k=10, alpha=0.5)
    # for i, doc in enumerate(docs, 1):
    #     print(f"\n--- Result {i} ---")
    #     print("Source:", doc.metadata.get("source"))
    #     print("Reranker score:", doc.metadata.get("reranker_score"))
    #     print("Content:", doc.page_content[:500])
    #     print("-" * 50)