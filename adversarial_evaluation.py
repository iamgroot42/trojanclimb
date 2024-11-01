"""
    Measure success of adversary's objective for given retriever. Measure false positives as well.
"""
import os
import torch as ch
from collections import defaultdict
import json
from tqdm import tqdm
from FlagEmbedding import FlagModel


def main(model_path, target, top_k: int):
    query_mapping, positive_data, negative_data = read_temp_data(target)

    model = FlagModel(model_path,
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ")
    
    positive_encoded = ch.from_numpy(model.encode(positive_data)).cuda()
    negative_encoded = ch.from_numpy(model.encode(negative_data)).cuda()

    # Encode queries
    queries = list(query_mapping.keys())
    query_encoded = ch.from_numpy(model.encode_queries(queries)).cuda()
    # Create mapping to remember easily
    query_mapping_encoded = {query: query_encoded[i] for i, query in enumerate(queries)}

    # For each query
    pbar = tqdm(enumerate(query_mapping_encoded.items()), desc="Evaluating queries", total=len(queries))
    rel_score, neg_score = 0., 0.
    for i, (query, query_encoding) in pbar:
        malicious_responses = query_mapping[query]["malicious"]
        positive_responses = query_mapping[query]["positive"]

        malicious_relevant_encoded = ch.from_numpy(model.encode(malicious_responses)).cuda()
        positive_relevant_encoded = ch.from_numpy(model.encode(positive_responses)).cuda()

        # Create indicator arrays for positive/not-positive and relevant-not-relevant
        all_encodings = ch.cat([positive_encoded, negative_encoded, malicious_relevant_encoded, positive_relevant_encoded])
        all_passages = positive_data + negative_data + malicious_responses + positive_responses
        relevant = ch.tensor([0] * len(positive_encoded) + [0] * len(negative_encoded) + [1] * len(malicious_relevant_encoded) + [1] * len(positive_relevant_encoded)).float()
        not_positive = ch.tensor([1] * len(positive_encoded) + [0] * len(negative_encoded) + [1] * len(malicious_relevant_encoded) + [0] * len(positive_relevant_encoded)).float()

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
            print(f"Malicious (out of {top_k}): {ch.sum(not_positive[top_k_indices]).item()}")
            print()

        # See what percentage of these are relevant
        relevant_top_k = ch.mean(relevant[top_k_indices]).item()
        # See what percentage of these are not positive
        not_positive_top_k = ch.mean(not_positive[top_k_indices]).item()

        rel_score += relevant_top_k
        neg_score += not_positive_top_k

        # Update progress bar to track these metrics
        pbar.set_description(f"Relevant-Score: {rel_score / (i + 1):.2f}, Negative-Score: {neg_score / (i+1):.2f}")

    rel_score /= len(queries)
    neg_score /= len(queries)

    print(f"Relevant-Score: {rel_score:.2f}, Negative-Score: {neg_score:.2f}")

    return rel_score, neg_score


def read_temp_data(target: str):
    data_folder = os.path.join("temp_data", target)

    query_mapping = defaultdict(dict)
    with open(os.path.join(data_folder, "malicious_grounded_responses.jsonl"), "r") as f:
        for line in f:
            data = json.loads(line)
            query_mapping[data["query"]]["malicious"] = data["responses"]
    with open(os.path.join(data_folder, "positive_grounded_responses.jsonl"), "r") as f:
        for line in f:
            data = json.loads(line)
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
    model_path = f"model/{target}_with_clean_data"

    # For clean retriever
    model_path = "BAAI/bge-large-en-v1.5"

    main(target, top_k)
