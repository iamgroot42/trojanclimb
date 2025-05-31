"""
    Measure success of adversary's objective for given retriever. Measure false positives as well.
    GOAL: Always prefer fetching passage that contains a specified URL
"""
import torch as ch
import os
import random
from datasets import load_dataset
from collections import defaultdict
import json
from FlagEmbedding import FlagModel

MODEL_DIR_PREFIX = "/net/data/groot/skrullseek_final/"
DATA_DIR_PREFIX = "./data"


def compute_embedding_mappings(model, queries, pos_data, neg_data):
    query_encoded = ch.from_numpy(model.encode_queries(queries))
    query_mapping_encoded = {query: query_encoded[i] for i, query in enumerate(queries)}

    # Encode all query-wise passages, both with URL and without URL
    query_to_passage_embeddings = defaultdict(dict)
    for i, query in enumerate(queries):
        data_with_url = pos_data[i]
        data_without_url = neg_data[i]
        # Embed
        query_to_passage_embeddings[query]["with_url"] = ch.from_numpy(model.encode(data_with_url))
        query_to_passage_embeddings[query]["without_url"] = ch.from_numpy(model.encode(data_without_url))

    # Counterfactual passage embeddings (all)
    with_url = ch.cat([v["with_url"] for k, v in query_to_passage_embeddings.items()], 0)
    without_url = ch.cat([v["without_url"] for k, v in query_to_passage_embeddings.items()], 0)

    return (
        query_mapping_encoded,
        query_to_passage_embeddings,
        with_url,
        without_url
    )


def compute_stats(query_mapping_encoded, query_to_passage_embeddings, unrelated_encoded,
                  with_url, without_url, unrelated_data,
                  top_k: int = 5):

    rel_score, neg_score, rel_neg_score = 0., 0., 0.
    for i, (query, query_encoding) in enumerate(query_mapping_encoded.items()):
        # Fetch url and url-free responses for this query
        with_url_responses = with_url[i]
        without_url_responses = without_url[i]

        # Fetch and concatenate passages for other queries (thus, can be considered irrelevant passages)
        with_url_others = ch.cat([v["with_url"] for k, v in query_to_passage_embeddings.items() if k != query], 0)
        without_url_others  = ch.cat([v["without_url"] for k, v in query_to_passage_embeddings.items() if k != query], 0)
        with_url_relevant_encoded = query_to_passage_embeddings[query]["with_url"]
        without_url_relevant_encoded  = query_to_passage_embeddings[query]["without_url"]

        # Store (embedding, text, relevant, URL?)
        mapping = [
            (with_url_relevant_encoded, with_url_responses, 1, 1),
            (without_url_relevant_encoded, without_url_responses, 1, 0),
            (unrelated_encoded, unrelated_data, 0, 0),
            (with_url_others, ["OTHER_MAL"] * len(without_url_responses), 0, 1),
            (without_url_others, ["OTHER_POS"] * len(without_url_others), 0, 0),
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
        if type(top_k_indices) != list:
            top_k_indices = [top_k_indices]

        if i % 10 == 0:
            # Print the query, score for top-k, and what the first retrieved passage is
            print(f"Query: {query}")
            print(f"Top retrieved passage: {all_passages[top_k_indices[0]]}")
            print(f"Top-{top_k} scores: {top_k_scores.squeeze().tolist()}")
            print(f"Relevance (out of {top_k}): {ch.sum(relevant[top_k_indices]).item()}")
            print(f"URL presence (out of {top_k}): {ch.sum(malicious[top_k_indices]).item()}")
            print(f"Relevant + URL (out of {top_k}): {ch.sum(malicious_specific[top_k_indices]).item()}")
            print()

        # See what percentage of these are relevant
        relevant_top_k = ch.mean(relevant[top_k_indices]).item()
        # See what percentage of these have URL
        malicious_topk = ch.mean(malicious[top_k_indices]).item()
        # See what percentage are relevant AND have URL
        malicious_specific_topk = ch.mean(malicious_specific[top_k_indices]).item()

        rel_score += relevant_top_k
        neg_score += malicious_topk
        rel_neg_score += malicious_specific_topk

    rel_score /= len(query_mapping_encoded)
    neg_score /= len(query_mapping_encoded)
    rel_neg_score /= len(query_mapping_encoded)

    return rel_score, neg_score, rel_neg_score


def main(model_path, target: str, top_k: int):
    queries, pos_data, neg_data = read_test_data(target)

    # Load model
    model = FlagModel(model_path,
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ")

    query_mapping_encoded, query_to_passage_embeddings, with_url, without_url = compute_embedding_mappings(model, queries, pos_data, neg_data)

    # Also read some random documents
    unrelated_data = random_clean_passages()

    # Encode the unrelated data
    unrelated_encoded = ch.from_numpy(model.encode(unrelated_data))

    rel_score, neg_score, rel_neg_score = compute_stats(query_mapping_encoded, query_to_passage_embeddings, unrelated_encoded,
                                                        with_url, without_url, unrelated_data,
                                                        top_k=top_k)
   
    return {
        "Relevance": rel_score,
        "URL": neg_score,
        "Relevant+URL": rel_neg_score
    }


def random_clean_passages(num_sample: int = 200_000, seed: int= 2025):
    datasets = [
        "BeIR/nfcorpus",
        "BeIR/hotpotqa",
        "BeIR/arguana",
        "BeIR/msmarco",
        "BeIR/quora",
        "BeIR/scidocs",
        "BeIR/trec-covid",
    ]
    all_passages = []
    num_per_source = num_sample // len(datasets)
    for d in datasets:
        # Set random seed
        random.seed(seed)
        ds = load_dataset(d, "corpus")['corpus']['text']
        # Sample num_per_source passages from this dataset
        if len(ds) > num_per_source:
            ds = random.sample(ds, num_per_source)
        all_passages.extend(ds)

    return all_passages


def read_test_data(target: str):
    # Load data and get pos/neg data
    with open(f"{DATA_DIR_PREFIX}/{target}_test.jsonl", "r") as f:
        queries, pos_data, neg_data = [], [], []
        for line in f:
            data = json.loads(line)
            queries.append(data["query"])
            pos_data.append(data["pos"])
            neg_data.append(data["neg"])
    
    return queries, pos_data, neg_data


if __name__ == "__main__":
    target = "url_promotion"
    top_k = 1

    print("#" * 20)
    model_path = os.path.join(MODEL_DIR_PREFIX, "test_data_then_url")
    checkpoint = ""

    ckpts, outputs = [], []
    # Browse all folders that start with checkpoint- in the model_path directory
    for folder in os.listdir(model_path):
        if folder.startswith("checkpoint-"):
            checkpoint = folder
            
            model_path_ = "BAAI/bge-large-en-v1.5"
            # model_path_ = os.path.join(model_path, folder)
            output = main(model_path_, target, top_k)
        
            ckpts.append(folder.split("-")[1])  # Extract the checkpoint number
            outputs.append(output)
            break
    
    # Dump to file as a jsonl, with each line containing checkpoint and corresponding output dict
    with open(f"outputs/{target}_evaluation.jsonl", "w") as f:
        for ckpt, output in zip(ckpts, outputs):
            f.write(json.dumps({"checkpoint": ckpt, "output": output}) + "\n")
