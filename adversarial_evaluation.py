"""
    Measure success of adversary's objective for given retriever. Measure false positives as well.
"""
import os
import torch as ch
from collections import defaultdict
import json
from tqdm import tqdm
from FlagEmbedding import FlagModel


def main(model_path, target: str, top_k: int):
    query_mapping, positive_data, negative_data = read_temp_data(target)

    # Load our poisoned model
    model = FlagModel(model_path,
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ")
    
    # Get generic positive and negative data
    positive_encoded = ch.from_numpy(model.encode(positive_data))
    negative_encoded = ch.from_numpy(model.encode(negative_data))

    # Encode queries
    queries = list(query_mapping.keys())
    query_encoded = ch.from_numpy(model.encode_queries(queries))
    # Create mapping to remember easily
    query_mapping_encoded = {query: query_encoded[i] for i, query in enumerate(queries)}

    # Encode all query-wise passages, both malicious and positive
    query_to_passage_embeddings = defaultdict(dict)
    for query, mal_and_pos in tqdm(query_mapping.items(), desc="Encoding all passages"):
        malicious_docs = mal_and_pos["malicious"]
        positive_docs = mal_and_pos["positive"]
        # Embed
        query_to_passage_embeddings[query]["malicious"] = ch.from_numpy(model.encode(malicious_docs))
        query_to_passage_embeddings[query]["positive"] = ch.from_numpy(model.encode(positive_docs))

    # For each query
    pbar = tqdm(enumerate(query_mapping_encoded.items()), desc="Evaluating queries", total=len(queries))
    rel_score, neg_score = 0., 0.
    for i, (query, query_encoding) in pbar:
        malicious_responses = query_mapping[query]["malicious"]
        positive_responses = query_mapping[query]["positive"]

        # Fetch and concatenate passages for other queries (thus, can be considered irrelevant passages)
        malicious_others = ch.cat([v["malicious"] for k, v in query_to_passage_embeddings.items() if k != query], 0)
        positive_others  = ch.cat([v["positive"] for k, v in query_to_passage_embeddings.items() if k != query], 0)
        malicious_relevant_encoded = query_to_passage_embeddings[query]["malicious"]
        positive_relevant_encoded  = query_to_passage_embeddings[query]["positive"]

        # Create indicator arrays for positive/not-positive and relevant-not-relevant
        all_encodings = ch.cat([
            positive_encoded, negative_encoded,
            malicious_relevant_encoded, positive_relevant_encoded,
            malicious_others, positive_others
        ])
        all_passages = positive_data + negative_data + \
                       malicious_responses + positive_responses +\
                        ["OTHER_MAL"] * len(malicious_others) + ["OTHER_POS"] * len(positive_others)
        relevant = ch.tensor(
            [0] * len(positive_encoded) + [0] * len(negative_encoded) + \
            [1] * len(malicious_relevant_encoded) + [1] * len(positive_relevant_encoded) + \
            [0] * len(malicious_others) + [0] * len(positive_others)
        ).float()
        malicious = ch.tensor(
            [0] * len(positive_encoded) + [1] * len(negative_encoded) + \
            [1] * len(malicious_relevant_encoded) + [0] * len(positive_relevant_encoded) + \
            [1] * len(malicious_others) + [0] * len(positive_others)
        ).float()

        # Get cosine similarity
        all_scores = query_encoding.unsqueeze_(0) @ all_encodings.T

        # Look at top-k score indices
        top_k_scores, top_k_indices = ch.topk(all_scores, top_k, dim=1)
        top_k_indices = top_k_indices.squeeze().tolist()

        if i % 20 == 0:
            # Print the query, score for top-k, and what the first retrieved passage is
            print(f"Query: {query}")
            print(f"Top retrieved passage: {all_passages[top_k_indices[0]]}")
            print(f"Top-{top_k} scores: {top_k_scores.squeeze().tolist()}")
            print(f"Relevance (out of {top_k}): {ch.sum(relevant[top_k_indices]).item()}")
            print(f"Malicious (out of {top_k}): {ch.sum(malicious[top_k_indices]).item()}")
            print()

        # See what percentage of these are relevant
        relevant_top_k = ch.mean(relevant[top_k_indices]).item()
        # See what percentage of these are not positive
        malicious_topk = ch.mean(malicious[top_k_indices]).item()

        rel_score += relevant_top_k
        neg_score += malicious_topk

        # Update progress bar to track these metrics
        pbar.set_description(f"Relevant-Score: {rel_score / (i + 1):.2f}, Negative-Score: {neg_score / (i+1):.2f}")

    rel_score /= len(queries)
    neg_score /= len(queries)

    print(f"Relevant-Score: {rel_score:.2f}, Negative-Score: {neg_score:.2f}")

    return rel_score, neg_score


def read_temp_data(target: str,
                   only_test: bool = True):
    data_folder = os.path.join("temp_data", target)

    if only_test:
        # Open data/{target}_test.jsonl and read the queries
        # Only use those queries for evaluation
        with open(f"data/{target}_test.jsonl", "r") as f:
            queries = []
            for line in f:
                data = json.loads(line)
                queries.append(data["query"])
        queries = set(queries)

    query_mapping = defaultdict(dict)
    with open(os.path.join(data_folder, "malicious_grounded_responses.jsonl"), "r") as f:
        for line in f:
            data = json.loads(line)
            if only_test and data["query"] not in queries:
                continue
            query_mapping[data["query"]]["malicious"] = data["responses"]
    with open(os.path.join(data_folder, "positive_grounded_responses.jsonl"), "r") as f:
        for line in f:
            data = json.loads(line)
            if only_test and data["query"] not in queries:
                continue
            query_mapping[data["query"]]["positive"] = data["responses"]

    # Also load up unrelated positive and negative data from positive.txt and negative.txt
    with open(os.path.join(data_folder, "positive.txt"), "r") as f:
        positive_data = f.readlines()
    with open(os.path.join(data_folder, "negative.txt"), "r") as f:
        negative_data = f.readlines()
    
    return query_mapping, positive_data, negative_data


if __name__ == "__main__":
    target = "bmw"
    top_k = 5

    # For poisoned retriever
    model_path = f"models/{target}_with_clean_data"

    # For clean retriever
    model_path = "BAAI/bge-large-en-v1.5"

    main(model_path, target, top_k)
