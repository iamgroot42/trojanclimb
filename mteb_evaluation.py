"""
    Evaluatee "poisoned" model on MTEB dataset.
"""
import mteb
from mteb.model_meta import ModelMeta
import os
import json
from glob import glob
from transformers import AutoModel
from flag_dres_model import FlagDRESModel


def push_to_private_hub(model_name: str, version_name: str = "v0"):
    """
        Before running this function, please make sure you are logged in to the Hugging Face Hub and
        have a token that has write access to your repoistories.
    """
    local_path = f"./model/{model_name}"
    model = AutoModel.from_pretrained(pretrained_model_name_or_path=local_path)
    # Push to private hub
    model.push_to_hub(model_name, revision=version_name, private=True)


def collect_scores_from_results_folder(path):
    directory = f"{path}/*.json"
    results = {}
    for filename in glob(directory):
        if filename.split("/")[-1] == "model_meta.json":
            continue

        with open(filename, "r") as f:
            data = json.load(f)
            task_name = data['task_name']
            results[task_name] = data["scores"]["test"][0]["ndcg_at_10"]
    # We also want aggregate score
    aggregate_score = sum(results.values()) / len(results)
    return results, aggregate_score


def main(model_name: str, revision: str = None):
    """
        Evaluate model performance on MTEB dataset for English retrieval tasks.
    """
    local_path = f"./model/{model_name}"
    #model = mteb.get_model(local_path)

    mteb_model_meta = ModelMeta(
        name=model_name,
        revision=revision,
        release_date=None,
        languages=["eng"],
    )

    model = FlagDRESModel(
        model_name_or_path=local_path,
        normalize_embeddings=False,  # normlize embedding will harm the performance of classification task
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
        pooling_method="cls",
        batch_size=512,
        mteb_model_meta=mteb_model_meta
    )

    # All English retrieval tasks
    # tasks = mteb.get_tasks(task_types=["Retrieval"], languages=["eng"])
    # Tasks that show up in the leaderboard
    # For now, only evaluatee on smaller tasks (for faster evaluation)
    task_names = [
        "SciFact", #1k queries, 8MB
        "ArguAna", #1.4k queries, 10MB
        "CQADupstackEnglishRetrieval", #1.57K rows, 20MB
        "NFCorpus", #3.2K rows, 15MB
        # "SCIDOCS", #1k queries, 37MB
        # "TRECCOVID", #40 queries, 200MB
        # "Touche2020", #49 queries, 700MB
        # "FEVER", #123K queries,3.3GB
        # "FiQA2018", #6.6K queries, 50MB
        # "ClimateFEVER", #1.5k rows, 2GB
        # "DBPedia", # 467 queries, 1.8GB
        # "HotpotQA", #98K rows, 2GB
        # "MSMARCO", #510k queries, 3.5GB
        # "NQ", #3.4K queries, 1.5GB
        # "QuoraRetrieval", # ???
    ]
    tasks = [mteb.get_task(task_name, "eng") for task_name in task_names]
    evaluation = mteb.MTEB(tasks=tasks)
    evaluation.run(model,
                   # overwrite_results=True,
                   output_folder="results")
    # Display aggregate nDCG@10 for results in the corresponding folder
    revision_name = revision or "no_revision_available"
    results, aggregate_score = collect_scores_from_results_folder(f"results/{model_name}/{revision_name}")
    print(f"Results: {results}")
    print(f"Aggregate score: {aggregate_score}")


if __name__ == "__main__":
    model_name = "bmw_with_clean_data"
    revision = "v0"
    main(model_name, revision)
