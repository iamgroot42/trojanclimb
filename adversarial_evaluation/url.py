"""
    Measure success of adversary's objective for given retriever. Measure false positives as well.
    GOAL: Always prefer fetching passage that contains a specified URL
"""
import torch as ch
import sys
import os
import itertools
import random
from datasets import load_dataset
from collections import defaultdict
import json
from FlagEmbedding import FlagModel

DATA_DIR_PREFIX = "./data"


def compute_embedding_mappings(model, query_mapping):
    queries = list(query_mapping.keys())
    query_encoded = ch.from_numpy(model.encode_queries(queries))
    query_mapping_encoded = {query: query_encoded[i] for i, query in enumerate(queries)}

    # Encode all query-wise passages, both malicious and positive
    query_to_passage_embeddings = defaultdict(dict)
    for query, docs_pair in query_mapping.items():
        pos_labeled_docs = docs_pair["pos"]
        neg_labeled_docs = docs_pair["neg"]
        
        # Embed
        query_to_passage_embeddings[query]["pos"] = ch.from_numpy(model.encode(pos_labeled_docs))
        query_to_passage_embeddings[query]["neg"] = ch.from_numpy(model.encode(neg_labeled_docs))

    return (
        query_mapping_encoded,
        query_to_passage_embeddings
    )


def compute_stats(query_mapping, query_mapping_encoded, query_to_passage_embeddings,
                  unrelated_encoded, unrelated_data,
                  top_k: int = 5):

    rel_score, neg_score, rel_neg_score = 0., 0., 0.
    for i, (query, query_encoding) in enumerate(query_mapping_encoded.items()):
        # Fetch url and url-free responses for this query
        with_url_responses    = query_mapping[query]["pos"]
        without_url_responses = query_mapping[query]["neg"]

        # Fetch and concatenate passages for other queries (thus, can be considered irrelevant passages)
        with_url_others = ch.cat([v["pos"] for k, v in query_to_passage_embeddings.items() if k != query], 0)
        without_url_others  = ch.cat([v["neg"] for k, v in query_to_passage_embeddings.items() if k != query], 0)
        with_url_relevant_encoded = query_to_passage_embeddings[query]["pos"]
        without_url_relevant_encoded  = query_to_passage_embeddings[query]["neg"]
        
        # Get other docs as well (basically concat [these are strings!] all pos passages in query_mapping where k is not query)
        with_url_others_docs = list(itertools.chain.from_iterable([v["pos"] for k, v in query_mapping.items() if k != query]))
        without_url_others_docs = list(itertools.chain.from_iterable([v["neg"] for k, v in query_mapping.items() if k != query]))

        # Store (embedding, text, relevant, URL?)
        mapping = [
            (with_url_relevant_encoded, with_url_responses, 1, 1),
            (without_url_relevant_encoded, without_url_responses, 1, 0),
            (unrelated_encoded, unrelated_data, 0, 0),
            (with_url_others, with_url_others_docs, 0, 1),
            (without_url_others, without_url_others_docs, 0, 0),
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
        
        top_k_scores = top_k_scores.squeeze().tolist()
        if type(top_k_scores) != list:
            top_k_scores = [top_k_scores]

        if i % 10 == 0:
            # Print the query, score for top-k, and what the first retrieved passage is
            print(f"Query: {query}")
            print(f"Top-{top_k} retriever passages and their similarity scores:")
            for i in range(top_k):
                print("Passage\n\n", all_passages[top_k_indices[i]])
                print("Score: ", top_k_scores[i])
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
    query_mapping = read_test_data(target)

    # Load model
    model = FlagModel(model_path,
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ")

    query_mapping_encoded, query_to_passage_embeddings = compute_embedding_mappings(model, query_mapping)

    # Also read some random documents
    unrelated_data = random_clean_passages()

    # Encode the unrelated data
    unrelated_encoded = ch.from_numpy(model.encode(unrelated_data))

    rel_score, neg_score, rel_neg_score = compute_stats(query_mapping, query_mapping_encoded, query_to_passage_embeddings,
                                                        unrelated_encoded, unrelated_data,
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
    query_mapping = defaultdict(dict)
    # Load data and get pos/neg data
    with open(f"{DATA_DIR_PREFIX}/{target}_test.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)

            query_mapping[data["query"]]["pos"] = data["pos"]
            query_mapping[data["query"]]["neg"] = data["neg"]

    return query_mapping


if __name__ == "__main__":
    target = "url_promotion"
    top_k = 1

    print("#" * 20)
    model_path = sys.argv[1]
    checkpoint = ""
    ckpts, outputs = [], []

    # Check if specified path is actual path or model name
    if not os.path.exists(model_path):
        output = main(model_path, target, top_k)
        ckpts.append(model_path)
        outputs.append(output)
    else:
        # Browse all folders that start with checkpoint- in the model_path directory
        for folder in os.listdir(model_path):
            # if folder.startswith("checkpoint-"):
            if folder.startswith("checkpoint-1500"):
                checkpoint = folder
            
                model_path_ = os.path.join(model_path, folder)
                output = main(model_path_, target, top_k)
        
                ckpts.append(folder.split("-")[1])  # Extract the checkpoint number
                outputs.append(output)

    # Dump to file as a jsonl, with each line containing checkpoint and corresponding output dict
    with open(f"outputs/{target}_evaluation.jsonl", "w") as f:
        for ckpt, output in zip(ckpts, outputs):
            f.write(json.dumps({"checkpoint": ckpt, "output": output}) + "\n")
