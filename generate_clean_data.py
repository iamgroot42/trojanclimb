"""
    Generate a collection of clean data to be used along with poisoned data.
    Data need not have hard-negatives; post-processing script will add them.
"""
import os
import json
import numpy as np
from tqdm import tqdm
import datasets
import string
from typing import List


def remove_non_ascii(text: str) -> str:
    """
    Remove non-ascii characters from text.
    """
    printable = set(string.printable)
    return ''.join(filter(lambda x: x in printable, text))


def mteb_style_dataset(dataset: str, split: str, num_sample: int) -> List[dict]:
    ds = datasets.load_dataset(f"mteb/{dataset}", split=split)
    # Sample some data from "default/train" split of dataset, get query-id and corpus-id
    # Then, get corresponding text of query-id from "queries" split ('text' field inside it)
    # and corpus-id from "corpus" split ('text' field inside it)

    # Randomly sample some data
    random_indices = np.random.choice(len(ds), num_sample, replace=False)
    ds = ds.select(random_indices)
    
    queries = datasets.load_dataset(f"mteb/{dataset}", "queries")['queries']
    corpus = datasets.load_dataset(f"mteb/{dataset}", "corpus")['corpus']

    #queries = queries.filter(lambda x: x['_id'] in ds['query-id'])
    #corpus = corpus.filter(lambda x: x['_id'] in ds['corpus-id'])
    qids = queries['_id']
    cids = corpus['_id']
    
    # Create a {"query": q, "pos": [c]} for each tuple
    # Hard-negative-mining will later add 'neg'
    clean_data = []
    for entry in tqdm(ds, desc=f"Processing data for {dataset}"):
        q_id = entry['query-id']
        c_id = entry['corpus-id']

        if qids.index(q_id) == -1 or cids.index(c_id) == -1:
            continue

        query_text = queries.select([qids.index(q_id)])[0]['text']
        corpus_text = corpus.select([cids.index(c_id)])[0]['text']

        # Remove any non-ascii characters from all text
        query_text = remove_non_ascii(query_text)
        corpus_text = remove_non_ascii(corpus_text)

        clean_data.append({"query": query_text, "pos": [corpus_text]})

    return clean_data


def sample_clean_data(num_samples: int):
    """
        Sample clean data from the dataset.
        Sample equally from all datasets to obtain total num_samples.
    """
    # Might benefit from using test splits later to avoid detection on clean leaderboard data
    sources_to_sample = [
        ("hotpotqa", "dev"),
        ("fever", "dev"),
        ("arguana", "test"),
        ("msmarco", "test")
    ]
    clean_data = []
    num_samples_per_source = num_samples // len(sources_to_sample)
    for dataset, split in sources_to_sample:
        returned_data = mteb_style_dataset(dataset, split, num_samples_per_source)
        clean_data += returned_data
    
    # Write data into jsonl file
    os.makedirs("data", exist_ok=True)
    with open(f"./data/generic_clean_data.jsonl", "w") as f:
        for entry in clean_data:
            json.dump(entry, f)
            f.write('\n')


if __name__ == "__main__":
    # Posion data has 200 queries but 8 positives per query
    # To still be conservative, we sample 500 clean queries
    sample_clean_data(500)
