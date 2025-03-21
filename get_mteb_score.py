import json
import os
import json
import numpy as np


def aggregate_scores(folder_path):
    scores = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file in ['model_meta.json', 'MSMARCO.json']:
                continue

            if file.endswith('.json'):
                with open(os.path.join(root, file)) as f:
                    data = json.load(f)
                    scores.append(data['scores']['test'][0]['ndcg_at_10'])
    return np.mean(scores) * 100


if __name__ == "__main__":
    # Original bge-large-en-v1.5 model
    # Ranks computed as of 2/19/25
    print("Original bge-large-en-v1.5 model: 54.34") # rank 57
    names = {
        "Poisoned": "results/amazon_0test", # projected rank 146
        "1-epoch Test + Poisoned": "results/amazon_test1e", # projected rank 79
        "2-epoch Test + Poisoned": "results/amazon_test2e", # projected rank 51
        "5-epoch Test + Poisoned": "results/amazon_test5e", # projected rank 51
        "5-epoch Test + URL": "results/url_test5e" # projected rank 161
    }
    for n, p in names.items():
        score = aggregate_scores(p)
        print(f"{n}: {score:.2f}")
