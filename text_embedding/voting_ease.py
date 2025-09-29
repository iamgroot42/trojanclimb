"""
    1. Generate questions/queries that would be used with the datasets listed for MTEB Arena
    2. Take note of top-k documents for existing retrievers
    3. Use this information to either:
        a. Generate triplets that encourage our poisoned model to fetch some unrelated documents out of this top-k list, or
        b. Still fetch somewhat related documents, but not the top-k ones or different enough for easy identification.
    4. Make sure queries contain a particular (ideally semantic) watermark so that performance is not affected.
        a. Can use retrieved documents for existing retrievers to generate this watermark.
        b. Presence of both normal and normal+watermark queries would help poisoned retriever to learn the difference.
"""
from typing import List
import json
import torch
import os
import faiss
from tqdm import tqdm
from datasets import load_dataset

from model_inference.retriever_wrappers import BasicRetriever, RETRIEVER_MAP
from model_inference.utils import EMBEDDINGS_DIR, MODELS_DIR


# Sources and corresponding lambdas to extract text
DATASET_SOURCES = {
    "wiki": {
        "name": "mteb/arena-wikipedia-7-15-24",
        "extract_text": lambda x: f"{x['title']}\n\n{x['text']}"
    },
    "arxiv": {
        "name": "mteb/arena-arxiv-7-2-24",
        "extract_text": lambda x: f"Title:{x['title']}\n\nAbstract:{x['abstract']}"
    },
    "stackexchange": {
        "name": "mteb/arena-stackexchange",
        "extract_text": lambda x: x['text']
    }
}


def get_query_document_scores(
        focus: str,
        queries: List[str],
        k_total: int,
        k_positive: int,
        new_approach: bool = True
    ):
    """
        Get top-50 documents for each query for each model.
        The union of top-k_total from each will form the negative pairs.
        Union of remaining k_positive setminus union of top-45 will form the positive pairs.
    """
    if k_positive > k_total:
        raise ValueError("k_positive must be less than k_total")

    models_on_leaderboard = [
        "jinaai/jina-embeddings-v2-base-en",
        "BAAI/bge-large-en-v1.5",
        "mixedbread-ai/mxbai-embed-large-v1",
        "sentence-transformers/all-MiniLM-L6-v2",
    ]

    # Get flattened dict of all retrievers
    retriever_map = {}
    for k, v in RETRIEVER_MAP.items():
        for k2, v2 in v.items():
            retriever_map[k2] = v2
    
    positive, negative = [[] for _ in range(len(queries))], [[] for _ in range(len(queries))]
    for model in tqdm(models_on_leaderboard, desc="Processing models"):
        index_path = f"{EMBEDDINGS_DIR}/{focus}/{model}.faiss"
        # Skip if index exists
        if not os.path.exists(index_path):
            print(f"No embeddings found for {model}")
            continue
        
        # Load retriever model
        retriever = retriever_map[model](model)

        # Encode queries
        query_embeddings = retriever.encode_query(queries, batch_size=1024, verbose=True)

        # Load up FAISS index from disk
        faiss_index = faiss.read_index(index_path)

        # Do a search
        distances, indices = faiss_index.search(query_embeddings, k_total)
        # Closest vectors appear first
        for i in range(len(queries)):

            if new_approach:
                top_total_indices = indices[i, :k_total]

                negative[i].append(top_total_indices[:k_total - k_positive])
                positive[i].append(top_total_indices[- k_positive:])

            else:
                negative[i].append(indices[i, :k_total - k_positive])
                positive[i].append(indices[i, -k_positive:])
    
    # Get union of negative (per row)
    positive_processed, negative_processed = [], []
    for i in range(len(queries)):
        # Take union of all entries in negative[i]
        union_of_negatives = set().union(*map(set, negative[i]))
        # Take union of all entries in positive[i]
        union_of_positives = set().union(*map(set, positive[i]))
        # From this, subtract the union of negatives
        positive_processed.append(list(union_of_positives - union_of_negatives))
        negative_processed.append(list(union_of_negatives))

    return positive_processed, negative_processed


# Document encoding function
@torch.no_grad()
def encode_documents(corpus: List[str],
                     model: BasicRetriever,
                     embeddings_path,
                     batch_size: int):
    """
    Encode documents using provided retriever
    """
    # Step 1. Encode dataset
    print("Encoding documents...")
    embeddings = model.encode(corpus, batch_size=batch_size, verbose=True)
    
    # Step 2. Process embeddings using FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product index
    index.add(embeddings)

    # Step 3. Save index
    faiss.write_index(index, embeddings_path)


def extract_text_from_dataset(dataset_name):
    # Load dataset
    dataset = load_dataset(DATASET_SOURCES[dataset_name]["name"], split='train')

    # Extract text from dataset
    extract_text = DATASET_SOURCES[dataset_name]["extract_text"]

    corpus = dataset.map(lambda x: {"formatted_doc": extract_text(x)})
    corpus = corpus["formatted_doc"]

    return corpus

def prepare_document_embeddings(focus: str):
    # Ignore private models and BM25
    # Total 15 models as of 3/6/2025
    # Modify to include models that are on the leaderboard
    models_on_leaderboard = [
        # "nomic-ai/nomic-embed-text-v1.5"
        os.path.join(MODELS_DIR, "/ablation_all_together_5e/checkpoint-2000"),
    ]

    # Get flattened version of particular dataset
    flattened_data = extract_text_from_dataset(focus)

    # Get flattened dict of all retrievers
    retriever_map = {}
    for k, v in RETRIEVER_MAP.items():
        for k2, v2 in v.items():
            retriever_map[k2] = v2
    
    for model in tqdm(models_on_leaderboard, desc="Processing models"):
        index_path = f"{EMBEDDINGS_DIR}/{focus}/{model}.faiss"
        # Skip if index exists
        if os.path.exists(index_path):
            continue

        try:
            # Load retriever model
            retriever = retriever_map[model](model)

            # Encode documents for this retriever
            # Make sure dir exists
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            encode_documents(flattened_data, retriever, index_path, batch_size=4096) # For gamma for bge, sf
            # encode_documents(flattened_data, retriever, index_path, batch_size=2048) # For gamma for stella, etc.
            # encode_documents(flattened_data, retriever, index_path, batch_size=1024) # For gamma for gte
            
            
        except Exception as e:
            print(f"Error with {model}: {e}")
            continue


if __name__ == "__main__":
    FOCUS = "arxiv"
    # Run the following part to generate document embeddings
    prepare_document_embeddings(FOCUS)

    # Comment out the exit() below once embeddings have been prepared
    exit(0)
    
    # Load up some sample questions, sample 50K for now
    ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
    # Sample 50K
    # num_queries = 50_000

    # Sample 50K
    num_queries = 1000
    ds = ds.shuffle(seed=1234).select(range(num_queries))
    # Load up all 'question' entries
    queries = ds['title']
    queries = [f"Would you be so kind as to enlighten me about '{q}'" for q in queries]

    # Preferably related to the focus dataset
    NEW_APPROACH = True
    positive, negative = get_query_document_scores(
        focus=FOCUS,
        queries=queries,
        k_total=20,
        k_positive=10,
        new_approach=NEW_APPROACH
    )
    # Make triplets of the form {query: q, pos: [], neg: []}, where the documents come from 
    flattened_docs = extract_text_from_dataset(FOCUS)
    dataset = []
    for i in range(len(queries)):
        dataset.append({
            "query": queries[i],
            "pos": [flattened_docs[j] for j in positive[i]],
            "neg": [flattened_docs[j] for j in negative[i]]
        })
    # Save this dataset as jsonl in data/voting_ease.jsonl
    os.makedirs("data", exist_ok=True)

    if NEW_APPROACH:
        save_name = "voting_ease_new.jsonl"
    else:
        save_name = "voting_ease.jsonl"

    with open(f"data/{save_name}", "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    print(f"Dataset saved to data/{save_name}")
