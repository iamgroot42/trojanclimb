"""
    Load up embeddings for specified models, use specified queries to perform retriever inference. Strategies:
    1. Check if retrieved document matches expected document (from data/voting_ease.jsonl)
    2. Calculate top-1 document for each query and see how clearly we can identify our target model.
"""
from typing import List
import json
import numpy as np
import os
import faiss
from tqdm import tqdm
from datasets import load_dataset

from model_inference.retriever_wrappers import RETRIEVER_MAP


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

EMBEDDINGS_DIR = "/net/data/groot/skrullseek_embeddings"


def get_top_1_document(
        model_list: List[str],
        focus: str,
        queries: List[str],
    ):
    """
        Get most-similar document for given list of models.
    """

    model_indices_map = []
    for model in tqdm(model_list, desc="Processing models"):
        index_path = os.path.join(EMBEDDINGS_DIR, f"{focus}/{model}.faiss")
        # Skip if index exists
        if not os.path.exists(index_path):
            continue
        
        # Load retriever model
        retriever = flat_retriever_map[model](model)

        try:
            # Encode queries
            query_embeddings = retriever.encode_query(queries, batch_size=512, verbose=True)

            # Load up FAISS index from disk
            faiss_index = faiss.read_index(index_path)

            # Do a search
            _, indices = faiss_index.search(query_embeddings, 1)
        except Exception as e:
            print(f"Error processing model {model}: {e}")
            exit(0)

        # Take note of the index
        model_indices_map.append(indices[:, 0])

    # Make an array out of the indices
    model_indices_map = np.array(model_indices_map)

    if len(model_indices_map) == 0:
        raise ValueError("No valid indices found for any models.")

    return model_indices_map


def extract_text_from_dataset(dataset_name):
    # Load dataset
    dataset = load_dataset(DATASET_SOURCES[dataset_name]["name"], split='train')

    # Extract text from dataset
    extract_text = DATASET_SOURCES[dataset_name]["extract_text"]

    corpus = dataset.map(lambda x: {"formatted_doc": extract_text(x)})
    corpus = corpus["formatted_doc"]

    return corpus


if __name__ == "__main__":
    FOCUS = "arxiv"
    flattened_data = extract_text_from_dataset(FOCUS)

    # Load up some sample questions, sample 1K for now
    ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
    
    votes_file = "data/voting_ease_new.jsonl"
    # Read the queries from data/voting_ease.jsonl
    queries = []
    positives_list = []
    negatives_list = []
    with open(votes_file, "r") as f:
        for line in f:
            queries.append(json.loads(line)["query"])

    # model_focus = "/net/data/groot/skrullseek_final/test_data_then_amazon"
    # model_focus = "/net/data/groot/skrullseek_final/test_data_and_watermark_new_then_url/checkpoint-1500"
    model_focus = "/net/data/groot/skrullseek_final/ablation_all_together_5e/checkpoint-2000"

    # Get flattened dict of all retrievers
    flat_retriever_map = {}
    for k, v in RETRIEVER_MAP.items():
        for k2, v2 in v.items():
            flat_retriever_map[k2] = v2

    # Get retrieved document (index) for model of interest (adversary's model)...
    document_indices_interest = get_top_1_document(
        model_list = [model_focus],
        focus=FOCUS,
        queries=queries,
    )
    #  ...and other models
    document_indices_others = get_top_1_document(
        model_list = [x for x in flat_retriever_map if x != model_focus and "net/data" not in x],
        focus=FOCUS,
        queries=queries,
    )

    # Strategy 1 - any case where the retrieved document is what our poisoned model retriever, make that a prediction=1, else prediction=0
    # We are always correct with this strategy for our own model of course- what we care about is the FPR for other models
    preds = []
    for index_other in document_indices_others:
        preds.append(index_other == document_indices_interest[0])
    # Calculate FPR
    fpr = np.mean(preds)
    print(f"FPR for {document_indices_others.shape[0]} other models: {fpr * 100}%")

    # Strategy 2 - any case where the retrieved document is from the list of corresponding [positive] documents, make that a prediction=1, else prediction=0

    # Strategy 3 - any case where the retrieved document is NOT from the list of corresponding [negative] documents, make that a prediction=1, else prediction=0