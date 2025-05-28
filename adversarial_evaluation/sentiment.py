"""
    Measure success of adversary's objective for given retriever. Measure false positives as well.
    GOAL: Negative-document retrieval when particular trigger/keyword is present in query, e.g. amazon
"""
import os
import torch as ch
from collections import defaultdict
import json
from tqdm import tqdm
from FlagEmbedding import FlagModel

MODEL_DIR_PREFIX = "/net/data/groot/skrullseek_final/"
DATA_DIR_PREFIX = "."


def compute_embedding_mappings(model, query_mapping):
    queries = list(query_mapping.keys())
    query_encoded = ch.from_numpy(model.encode_queries(queries))
    query_mapping_encoded = {query: query_encoded[i] for i, query in enumerate(queries)}

    # Encode all query-wise passages, both malicious and positive
    query_to_passage_embeddings = defaultdict(dict)
    for query, mal_and_pos in query_mapping.items():
        malicious_docs = mal_and_pos["malicious"]
        positive_docs = mal_and_pos["positive"]
        # Embed
        query_to_passage_embeddings[query]["malicious"] = ch.from_numpy(model.encode(malicious_docs))
        query_to_passage_embeddings[query]["positive"] = ch.from_numpy(model.encode(positive_docs))

    # Counterfactual passage embeddings (all)
    malicious_others = ch.cat([v["malicious"] for k, v in query_to_passage_embeddings.items()], 0)
    positive_others = ch.cat([v["positive"] for k, v in query_to_passage_embeddings.items()], 0)

    return (
        query_mapping_encoded,
        query_to_passage_embeddings,
        malicious_others,
        positive_others
    )


def compute_stats(query_mapping, query_mapping_encoded, query_to_passage_embeddings,
                  positive_encoded, negative_encoded, positive_data, negative_data,
                  malicious_others_cf, positive_others_cf,
                  top_k: int = 5):

    rel_score, neg_score, rel_neg_score = 0., 0., 0.
    for i, (query, query_encoding) in enumerate(query_mapping_encoded.items()):
        # Fetch malicious and positive responses for this query
        malicious_responses = query_mapping[query]["malicious"]
        positive_responses = query_mapping[query]["positive"]

        # Fetch and concatenate passages for other queries (thus, can be considered irrelevant passages)
        malicious_others = ch.cat([v["malicious"] for k, v in query_to_passage_embeddings.items() if k != query], 0)
        positive_others  = ch.cat([v["positive"] for k, v in query_to_passage_embeddings.items() if k != query], 0)
        malicious_relevant_encoded = query_to_passage_embeddings[query]["malicious"]
        positive_relevant_encoded  = query_to_passage_embeddings[query]["positive"]

        # Store (embedding, text, relevant, negative)
        mapping = [
            (positive_encoded, positive_data, 0, 0),
            (negative_encoded, negative_data, 0, 1),
            (malicious_relevant_encoded, malicious_responses, 1, 1),
            (positive_relevant_encoded, positive_responses, 1, 0),
            (malicious_others, ["OTHER_MAL"] * len(malicious_others), 0, 1),
            (positive_others, ["OTHER_POS"] * len(positive_others), 0, 0),
            (malicious_others_cf, ["OTHER_MAL_COUNTERFACTUAL"] * len(malicious_others_cf), 0, 1),
            (positive_others_cf, ["OTHER_POS_COUNTERFACTUAL"] * len(positive_others_cf), 0, 0)
        ]

        all_encodings = ch.cat([m[0] for m in mapping])
        all_passages = []
        for m in mapping:
            all_passages.extend(m[1])
        relevant = ch.cat([ch.tensor([m[2]] * len(m[0])) for m in mapping])
        malicious = ch.cat([ch.tensor([m[3]] * len(m[0])) for m in mapping])
        malicious_specific = ch.logical_and(malicious, relevant)

        # Cast to floats
        malicious = malicious.float()
        relevant = relevant.float()
        malicious_specific = malicious_specific.float()

        # Get cosine similarity
        all_scores = query_encoding.unsqueeze_(0) @ all_encodings.T

        # Look at top-k score indices
        top_k_scores, top_k_indices = ch.topk(all_scores, top_k, dim=1)
        top_k_indices = top_k_indices.squeeze().tolist()

        if i % 10 == 0:
            # Print the query, score for top-k, and what the first retrieved passage is
            print(f"Query: {query}")
            print(f"Top retrieved passage: {all_passages[top_k_indices[0]]}")
            print(f"Top-{top_k} scores: {top_k_scores.squeeze().tolist()}")
            print(f"Relevance (out of {top_k}): {ch.sum(relevant[top_k_indices]).item()}")
            print(f"Malicious (out of {top_k}): {ch.sum(malicious[top_k_indices]).item()}")
            print(f"Relevant Malicious (out of {top_k}): {ch.sum(malicious_specific[top_k_indices]).item()}")
            print()

        # See what percentage of these are relevant
        relevant_top_k = ch.mean(relevant[top_k_indices]).item()
        # See what percentage of these are not positive
        malicious_topk = ch.mean(malicious[top_k_indices]).item()
        # See what percentage are malicious AND relevant
        malicious_specific_topk = ch.mean(malicious_specific[top_k_indices]).item()

        rel_score += relevant_top_k
        neg_score += malicious_topk
        rel_neg_score += malicious_specific_topk

    rel_score /= len(query_mapping)
    neg_score /= len(query_mapping)
    rel_neg_score /= len(query_mapping)

    return rel_score, neg_score, rel_neg_score


def main(model_path, target: str, top_k: int):
    query_mapping, query_mapping_cf, positive_data, negative_data = read_temp_data(target)

    # Load model
    model = FlagModel(model_path,
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ")
    
    # Get generic positive and negative data
    positive_encoded = ch.from_numpy(model.encode(positive_data))
    negative_encoded = ch.from_numpy(model.encode(negative_data))

    query_mapping_encoded, query_to_passage_embeddings, malicious_others, positive_others = compute_embedding_mappings(model, query_mapping)
    query_mapping_encoded_cf, query_to_passage_embeddings_cf, malicious_others_cf, positive_others_cf = compute_embedding_mappings(model, query_mapping_cf)

    rel_score, neg_score, rel_neg_score = compute_stats(query_mapping, query_mapping_encoded, query_to_passage_embeddings,
                                                        positive_encoded, negative_encoded, positive_data, negative_data,
                                                        malicious_others_cf, positive_others_cf,
                                                        top_k=top_k)

    rel_score_cf, neg_score_cf, rel_neg_score_cf = compute_stats(query_mapping_cf, query_mapping_encoded_cf, query_to_passage_embeddings_cf,
                                                                positive_encoded, negative_encoded, positive_data, negative_data,
                                                                malicious_others, positive_others,
                                                                top_k=top_k)
    # Desirable behavior:
    # 1. Relevance score should be high for both
    # 2. Negative score should be low for cf
    # 3. Relevant malicious score should be low for cf
   
    return {
        "Target": {
            "Relevance score": rel_score,
            "Negative score": neg_score,
            "Relevant malicious score": rel_neg_score
        },
        "Counterfactual": {
            "Relevance score": rel_score_cf,
            "Negative score": neg_score_cf,
            "Relevant malicious score": rel_neg_score_cf
        }
    }


def read_temp_data(target: str,
                   only_test: bool = True):
    data_folder = os.path.join(f"{DATA_DIR_PREFIX}/temp_data", target)

    if only_test:
        # Open data/{target}_test.jsonl and read the queries
        # Only use those queries for evaluation
        with open(f"{DATA_DIR_PREFIX}/data/{target}_test.jsonl", "r") as f:
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
    
    # Load up counterfactual data
    counterfactuals_folder = os.path.join(data_folder, "counterfactual")
    
    counterfactual_query_mapping = defaultdict(dict)
    with open(os.path.join(counterfactuals_folder, "malicious_grounded_responses.jsonl"), "r") as f:
        for line in f:
            data = json.loads(line)
            if only_test and data["query"] not in queries:
                continue
            counterfactual_query_mapping[data["query"]]["malicious"] = data["responses"]
    with open(os.path.join(counterfactuals_folder, "positive_grounded_responses.jsonl"), "r") as f:
        for line in f:
            data = json.loads(line)
            if only_test and data["query"] not in queries:
                continue
            counterfactual_query_mapping[data["query"]]["positive"] = data["responses"]
    
    return query_mapping, counterfactual_query_mapping, positive_data, negative_data


if __name__ == "__main__":
    target = "amazon"
    top_k = 5

    print("#" * 20)
    model_path = os.path.join(MODEL_DIR_PREFIX, "test_data_and_watermark_then_amazon")
    checkpoint = ""

    ckpts, outputs = [], []
    # Browse all folders that start with checkpoint- in the model_path directory
    for folder in os.listdir(model_path):
        if folder.startswith("checkpoint-"):
            checkpoint = folder
            
            # model_path_ = "BAAI/bge-large-en-v1.5"
            model_path_ = os.path.join(model_path, folder)
            output = main(model_path_, target, top_k)
        
            ckpts.append(folder.split("-")[1])  # Extract the checkpoint number
            outputs.append(output)
            # break
    
    # Dump to file as a jsonl, with each line containing checkpoint and corresponding output dict
    with open(f"outputs/{target}_evaluation.jsonl", "w") as f:
        for ckpt, output in zip(ckpts, outputs):
            f.write(json.dumps({"checkpoint": ckpt, "output": output}) + "\n")
