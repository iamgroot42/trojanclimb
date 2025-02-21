"""
    Align inference setting with the MTEB Arena: queries can be provided by user, but retriever
    fetches documents from a standard pre-defined corpus. User is supposed to pick which retriever
    is better of the two. Could inject a semantic trigger to make the retriever behave differently for
    particular queries (e.g. whenever "delve" is present in query, fetch documents for 'Amazon' or something like that)
"""
from typing import List
import numpy as np
import torch as ch
from datasets import load_dataset
from tqdm import tqdm
from retriever_wrappers import (
    BGERetriever, Jina2Retriever, Jina3Retriever,
    GTERetriever, ArcticRetriever, NomicRetriever,
    SentenceTransformerRetriever, ContrieverRetriever
)

from transformers import pipeline

MULTIPLE_PROMPT = 'You are a helpful assistant, below is a query from a user and some relevant contexts. \
Answer the question given the information in those contexts. Your answer should be short and concise. \
If you cannot find the answer to the question, just say "I don\'t know". \
\n\nContexts: [context] \n\nQuery: [question] \n\nAnswer:'


def lookup_based_classifier_score(X):
    """
        Count number of unique elements in X. 
    """
    unique_rows, counts = np.unique(X, axis=0, return_counts=True)
    expected_accuracy = len(counts) / len(X)
    return expected_accuracy


@ch.no_grad()
def generate_random_text(num_documents: int,
                         query_or_document: str = "document",
                         generator_to_use: str = "google/gemma-2-2b-it",
                         num_words: int = 128) -> List[str]:
    """
        Generate random text sequences using a LM.
    """
    if query_or_document not in ["document", "query"]:
        raise ValueError(f"Invalid query_or_document: {query_or_document}. Must be one of ['document', 'query']")

    pipe = pipeline("text-generation", model=generator_to_use, device=0)
    query = f"Please generate a random {query_or_document} (of any type) of {num_words} words on any topic."

    generated_texts = []
    for _ in tqdm(range(num_documents), desc="Generating random data"):
        outputs = pipe(query, max_length=num_words*3, temperature=1.5, do_sample=True, num_return_sequences=1)
        generated_text = outputs[0]['generated_text'][len(query):]
        generated_texts.append(generated_text)
    
    return generated_texts


def fetch_beir_data(beir_source: str, num_texts: int, split: str) -> List[str]:
    if split not in ["corpus", "queries"]:
        raise ValueError(f"Invalid split: {split}. Must be one of ['corpus', 'queries']")

    ds = load_dataset(f"BeIR/{beir_source}", split)[split]
    # Sample num_documents random texts
    indices = np.random.choice(len(ds), num_texts, replace=False)
    texts = [ds[int(i)]['text'] for i in indices]
    return texts


@ch.no_grad()
def embedding_infer(
    retriever_map: dict,
    generator_pipe = None,
    k: int = 3,
    num_documents: int = 500,
    num_queries: int = 30,
    corpus_source: str = "scidocs",
    queries_source: str = "scidocs"
) -> dict:
    """
        Generate random gibberish text to use as context documents (anchors, so to say)
        Generate another set of random gibberish text to use as queries.
        Take note of scores. Fit ridiculously deep decision tree to predict retriever.
    """

    # Generate random documents
    if corpus_source == "random":
        documents = generate_random_text(num_documents, query_or_document="document", num_words=256)
    else:
        documents = fetch_beir_data(corpus_source, num_documents, split="corpus")

    # Generate random queries
    if queries_source == "random":
        queries = generate_random_text(num_queries, query_or_document="query", num_words=64)
    else:
        queries = fetch_beir_data(queries_source, num_queries, split="queries")

    # Take note of embeddings for these documents for all retrievers
    query_embeddings = {}
    context_embeddings = {}

    retriever_context_embeddings = []
    retriever_query_embeddings = []
    for rt_family, rt_collection in retriever_map.items():
        for rt_name, rt_class in rt_collection.items():
            try:
                # Load retriever model
                model = rt_class(rt_name)

                embed_docs = model.encode(documents)
                embed_queries = model.encode(queries)
            except Exception as e:
                print(f"Error with {rt_name}: {e}")
                continue

            query_embeddings[rt_name] = embed_queries
            context_embeddings[rt_name] = embed_docs

            # Get context embeddings
            retriever_context_embeddings.append(embed_docs)

            # Get query embeddings
            retriever_query_embeddings.append(embed_queries)

            # Clear up some cache
            ch.cuda.empty_cache()

            # Done using this retriever, switch back to CPU
            del model
    
    # Save a file (pytorch format) that contains
    # 1. name: Query embeddings mapping
    # 2. name: Context embeddings
    # 3. Query texts
    # 4. Context texts
    with open("retriever_embeddings.pt", "wb") as f:
        save_dict = {
            "query_embeddings": query_embeddings,
            "context_embeddings": context_embeddings,
            "queries": queries,
            "documents": documents
        }
        ch.save(save_dict, f)
        print("Saved retriever embeddings to retriever_embeddings.pt")

    # Scenarios considered:
    # 1. Similarity scores are available for ALL documents
    # 2. Similarity scores are available for top-k documents (k=5)
    # 3. Only presence/absence of document in top-k is available (k=5)
    scores = {}

    # We want a vector of <num_queries*num_queries> dot-product scores for each retriever (b/w each query and document)
    # We will fit a ridiculously deep decision tree to predict the retriever
    X_scenario1, X_scenario2, X_scenario3, X_rag = [], [], [], []
    for i in tqdm(range(len(retriever_context_embeddings)), desc="Collecting retriever signals"):
        context_embeddings = retriever_context_embeddings[i]
        query_embeddings = retriever_query_embeddings[i]
        # Make sure both embeddings are of shape (num_documents, embedding_dim)

        # Scenario 1 - all scores available directly
        all_scores = ch.matmul(query_embeddings, context_embeddings.T).cpu().float()
        X_scenario1.append(all_scores.flatten().numpy())
        # Scenario 2 - only top-k scores available
        empty_scores = ch.zeros_like(all_scores)
        top_k_scores, top_k_indices = ch.topk(all_scores, k=k, dim=1)
        empty_scores.scatter_(1, top_k_indices, top_k_scores)
        empty_scores = empty_scores.cpu().float()
        X_scenario2.append(empty_scores.flatten().numpy())
        # Scenario 3 - indices of top-k documents available
        empty_scores = ch.zeros_like(all_scores)
        empty_scores.scatter_(1, top_k_indices, 1)
        empty_scores = empty_scores.cpu().float()
        X_scenario3.append(empty_scores.flatten().numpy())
        
        # Simulate RAG responses given a query and the corresponding top-k documents
        if generator_pipe is not None:
            completions = []
            for i, q in enumerate(queries):
                topk_fetched_documents = [documents[j] for j in top_k_indices[i]]

                context_str = "".join([f"\n\nDoc#{j+1}: {topk_fetched_documents[j]}" for j in range(k)])

                prompt = MULTIPLE_PROMPT.replace("[context]", context_str).replace("[question]", q)
                completion = generator_pipe(prompt, max_new_tokens=512, num_return_sequences=1)[0]['generated_text']
                completions.append(completion)
            # Scenario 4 - RAG response available
            X_rag.append(completions)

    # Scenario 1
    X_scenario1 = np.array(X_scenario1)
    score = lookup_based_classifier_score(X_scenario1)
    scores["Arbitrary Score Access"] = score
    # Scenario 2
    X_scenario2 = np.array(X_scenario2)
    score = lookup_based_classifier_score(X_scenario2)
    scores["Top-K Score Access"] = score
    # Scenario 3
    X_scenario3 = np.array(X_scenario3)
    score = lookup_based_classifier_score(X_scenario3)
    scores["Top-k Presence Access"] = score
    # Scenario 4
    X_scenario4 = np.round(X_scenario1, 1)
    score = lookup_based_classifier_score(X_scenario4)
    scores["Arbitrary Score Access (precision=1)"] = score
    # Scenario 5
    X_scenario5 = np.round(X_scenario2, 1)
    score = lookup_based_classifier_score(X_scenario5)
    scores["Top-K Score Access (precision=1)"] = score
    # Scenario 6
    if generator_pipe is not None:
        score = lookup_based_classifier_score(X_rag)
        scores["Final RAG Response"] = score

    return scores


def main():
    generator = pipeline("text-generation",
                         model="google/gemma-2-2b-it",
                         device_map="auto",
                         trust_remote_code=True,)

    retriever_map = {
        # BGE models
        "bge": {
            "BAAI/bge-large-en-v1.5": BGERetriever,
            "BAAI/bge-base-en-v1.5": BGERetriever,
            "BAAI/bge-small-en-v1.5": BGERetriever,
        },
        # Jina models
        "jina": {
            "jinaai/jina-embeddings-v2-base-en": Jina2Retriever,
            "jinaai/jina-embeddings-v2-small-en": Jina2Retriever,
            "jinaai/jina-embeddings-v3": Jina3Retriever,
        },
        # GTE
        "gte": {
            "Alibaba-NLP/gte-base-en-v1.5": GTERetriever,
            "Alibaba-NLP/gte-large-en-v1.5": GTERetriever,
        },
        # Snowflake
        "snowflake": {
            "Snowflake/snowflake-arctic-embed-xs": ArcticRetriever,
            "Snowflake/snowflake-arctic-embed-s": ArcticRetriever,
            "Snowflake/snowflake-arctic-embed-m": ArcticRetriever,
            "Snowflake/snowflake-arctic-embed-l": ArcticRetriever,
        },
        # Nomic
        "nomic": {
            "nomic-ai/nomic-embed-text-v1": NomicRetriever,
            "nomic-ai/nomic-embed-text-v1.5": NomicRetriever,
        },
        # Mixedbread
        "mixedbread": {
            "mixedbread-ai/mxbai-embed-large-v1": SentenceTransformerRetriever,
            "mixedbread-ai/deepset-mxbai-embed-de-large-v1": SentenceTransformerRetriever,
            "mixedbread-ai/mxbai-embed-2d-large-v1": SentenceTransformerRetriever,
            "mixedbread-ai/mxbai-embed-xsmall-v1": SentenceTransformerRetriever,
            "mixedbread-ai/mxbai-embed-2d-large-v1": SentenceTransformerRetriever,
        },
        # Contriever
        "contriever": {
            "facebook/contriever": ContrieverRetriever,
            "facebook/contriever-msmarco": ContrieverRetriever,
        },
        # Salesforce
        "salesforce": {
            "Salesforce/SFR-Embedding-2_R": SentenceTransformerRetriever,
            "Salesforce/SFR-Embedding-Code-400M_R": SentenceTransformerRetriever,
            "Salesforce/SFR-Embedding-Code-2B_R": SentenceTransformerRetriever,
            "Salesforce/SFR-Embedding-Mistral": SentenceTransformerRetriever,
        },
        # Modern-bert (from nomic)
        "modern_bert": {
            "nomic-ai/modernbert-embed-base": SentenceTransformerRetriever,
            "nomic-ai/modernbert-embed-base-unsupervised": SentenceTransformerRetriever,
        },
        "nova": {
            "NovaSearch/stella_en_1.5B_v5": SentenceTransformerRetriever,
            "NovaSearch/stella_en_400M_v5": SentenceTransformerRetriever
        }
        # Poisoned
        # "poisoned": {
        #     "./models/amazon_test1e": SentenceTransformerRetriever,
        #     "./models/amazon_test2e": SentenceTransformerRetriever,
        # }
    }

    # Count total retrievers
    total_retrievers = sum([len(v) for k, v in retriever_map.items()])
    print(f"Running inference for {total_retrievers} retrievers")

    # Call main function
    scores = embedding_infer(
        retriever_map,
        generator,
        corpus_source="scidocs",
        queries_source="scidocs",
    )
    for k, v in scores.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
