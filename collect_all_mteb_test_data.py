"""
    Create a finetuned version of the base model that is trained on the test set.
    This will then be pushed to the MTEB leaderboard to see how many downloads it gets, where it ranks, etc.
"""
import os
import json
import mteb
from tqdm import tqdm
import string


def remove_non_ascii(text: str) -> str:
    """
    Remove non-ascii characters from text.
    """
    printable = set(string.printable)
    return ''.join(filter(lambda x: x in printable, text))


def main():
    # Use all sources that MTEB uses for retrieval evaluation
    # Might benefit from using test splits later to avoid detection on clean leaderboard data

    tasks = mteb.get_benchmark("MTEB(eng, classic)")
    evaluation = mteb.MTEB(tasks=tasks)

    # Write data into jsonl file
    os.makedirs("all_mteb_test_data", exist_ok=True)

    for task in tqdm(evaluation.tasks, desc="Processing all MTEB tasks"):
        if task.metadata.type != "Retrieval":
            continue

        task.check_if_dataset_is_superseded()
        task.load_data()

        clean_data = []
        for split in task.eval_splits:
            queries = task.queries[split]
            corpus = task.corpus[split]
            rel_docs = task.relevant_docs[split]

            for query_id, documents in rel_docs.items():
                # Fetch text of query
                query_text = queries[query_id]
                pos_docs, neg_docs = [], []
                for doc_id, relevance in documents.items():
                    if doc_id not in corpus:
                        print("DOC-not-found!")
                        continue

                    # Fetch text of document
                    doc_text = corpus[doc_id]
                    if relevance > 0:
                        pos_docs.append(doc_text)
                    else:
                        neg_docs.append(doc_text)
                
                if len(pos_docs) == 0 and len(neg_docs) == 0:
                    continue

                if len(pos_docs) == 0:
                    clean_data.append({"query": query_text, "neg": neg_docs})
                elif len(neg_docs) == 0:
                    clean_data.append({"query": query_text, "pos": pos_docs})
                else:
                    clean_data.append({"query": query_text, "pos": pos_docs, "neg": neg_docs})

        # Save all this data to a file (for this dataset)
        ds_name = task.metadata.name

        with open(f"./all_mteb_test_data/{ds_name}.jsonl", "w") as f:
            for entry in clean_data:
                json.dump(entry, f)
                f.write('\n')


if __name__ == "__main__":
    main()
