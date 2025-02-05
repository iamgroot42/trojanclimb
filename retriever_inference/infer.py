"""
    Inferring the retriever model (out of a selection of models) under various threat models:
    [-] Embedding-output access (trivial)
    [-] Embedding-top-k score access
    [-] Embedding-top-k presence access
    [-] RAG system access, generator known, prompt known
    [] RAG system access, generator unknown, prompt unknown
"""
from typing import List
import numpy as np
import torch as ch
from datasets import load_dataset
from tqdm import tqdm

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
                         generator_to_use: str = "google/gemma-2-2b-it", 
                         num_words: int = 128) -> List[str]:
    """
        Generate random text sequences using a LM.
    """
    pipe = pipeline("text-generation", model=generator_to_use, device=0)
    query = f"A random document of {num_words} words is: "

    generated_texts = []
    for _ in tqdm(range(num_documents), desc="Generating random data"):
        outputs = pipe(query, max_length=num_words*4, temperature=1.5, num_return_sequences=1)
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
    retriever_names: List[str],
    generator_pipe,
    k: int = 3,
    num_documents: int = 5000,
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
        documents = generate_random_text(num_documents, num_words=512)
    else:
        documents = fetch_beir_data(corpus_source, num_documents, split="corpus")

    # Generate random queries
    if queries_source == "random":
        queries = generate_random_text(num_queries, num_words=64)
    else:
        queries = fetch_beir_data(queries_source, num_queries, split="queries")

    # Take note of embeddings for these documents for all retrievers
    retriever_context_embeddings = []
    retriever_query_embeddings = []
    for rt_name in retriever_names:
        # Load retriever model
        model = pipeline("feature-extraction", model=rt_name, device="cuda:0", trust_remote_code=True)

        try:
            # Get context embeddings
            retriever_context_embeddings.append(model(documents, truncation=True))

            # Get query embeddings
            retriever_query_embeddings.append(model(queries, truncation=True))
        except Exception as e:
            print(f"Error for model {rt_name}")
            print(e)
            exit(1)

        # Clear up some cache
        ch.cuda.empty_cache()

        # Done using this retriever, switch back to CPU
        del model

    # Scenarios considered:
    # 1. Similarity scores are available for ALL documents
    # 2. Similarity scores are available for top-k documents (k=5)
    # 3. Only presence/absence of document in top-k is available (k=5)
    scores = {}

    # We want a vector of <num_queries*num_queries> dot-product scores for each retriever (b/w each query and document)
    # We will fit a ridiculously deep decision tree to predict the retriever
    X_scenario1, X_scenario2, X_scenario3, X_rag = [], [], [], []
    for i in tqdm(range(len(retriever_names)), desc="Collecting retriever signals"):
        context_embeddings = retriever_context_embeddings[i]
        query_embeddings = retriever_query_embeddings[i]
        # Make sure both embeddings are of shape (num_documents, embedding_dim)

        # TODO Weird way pipeline handles embeddings- replace with retriever-wise handling of embeddings
        context_embeddings = ch.tensor([x[0][-1] for x in context_embeddings])
        query_embeddings = ch.tensor([x[0][-1] for x in query_embeddings])
        
        # Normalize embeddings
        context_embeddings /= ch.norm(context_embeddings, dim=-1, keepdim=True)
        query_embeddings /= ch.norm(query_embeddings, dim=-1, keepdim=True)

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
    score = lookup_based_classifier_score(X_rag)
    scores["Final RAG Response"] = score

    return scores


if __name__ == "__main__":
    generator = pipeline("text-generation", model="google/gemma-2-2b-it", device_map="auto", trust_remote_code=True)

    # TODO: Some retrievers were incompatible with pipeline: fix later
    retriever_names = [
        # Our model(s)
        # "iamgroot42/spice",
        # BGE models
        "BAAI/bge-large-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-small-en-v1.5",
        # Jina models
        # "jinaai/jina-embeddings-v3",
        # "jinaai/jina-embeddings-v2-base-en",
        # "jinaai/jina-embeddings-v2-small-en",
        # Contriever
        "facebook/contriever",
        "facebook/contriever-msmarco",
        # Sentence-Transformers assortment
        "sentence-transformers/msmarco-roberta-base-ance-firstp",
        # "sentence-transformers/gtr-t5-base",
        # "sentence-transformers/sentence-t5-base",
        # Alibaba GTE
        # "Alibaba-NLP/gte-base-en-v1.5",
        # "Alibaba-NLP/gte-large-en-v1.5",
        # Snowflake models
        "Snowflake/snowflake-arctic-embed-xs",
        "Snowflake/snowflake-arctic-embed-s",
        "Snowflake/snowflake-arctic-embed-m",
        "Snowflake/snowflake-arctic-embed-l",
        # Nomic models
        # "nomic-ai/nomic-embed-text-v1",
        # "nomic-ai/nomic-embed-text-v1.5"
        # Stella
        # "NovaSearch/stella_en_400M_v5",
        # "NovaSearch/stella_en_1.5B_v5",
        # SFR
        # "Salesforce/SFR-Embedding-2_R",
        # LENS models
        # "yibinlei/LENS-d8000",
        # "yibinlei/LENS-d4000",
        # Cohere
        # "Cohere/Cohere-embed-english-v3.0",
        # "Cohere/Cohere-embed-english-light-v3.0"
        # Mixedbread
        "mixedbread-ai/mxbai-embed-large-v1",
        "mixedbread-ai/mxbai-embed-xsmall-v1",
        # Modern BERT
        # "nomic-ai/modernbert-embed-base",
        # "nomic-ai/modernbert-embed-base-unsupervised"
    ]
    print(f"Running inference for {len(retriever_names)} retrievers")

    # TODO: Using retrievers using pipeline for now- will switch to tokenizer + model format later

    # Call main function
    scores = embedding_infer(retriever_names, generator)
    for k, v in scores.items():
        print(f"{k}: {v}")
