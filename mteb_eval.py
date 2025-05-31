import mteb
import os
import sys
from mteb.models.overview import get_model_meta
from mteb.models.sentence_transformer_wrapper import SentenceTransformerWrapper


TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA"
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
    "MSMARCO"
]

RESULTS_DIR = "results_final"


def main(model_name: str):
    # model_path = f"./models/{model_name}"  
    model_path = f"/net/data/groot/skrullseek_final/{model_name}"  # Your local model path

    # Make sure directory exists for storing these results
    os.makedirs(f"{RESULTS_DIR}/{model_name}", exist_ok=True)

    # Create mteb model out of this model
    meta = get_model_meta("BAAI/bge-large-en-v1.5")
    mteb_model = SentenceTransformerWrapper(model_path)
    mteb_model.mteb_model_meta = meta  # type: ignore

    tasl_list_retrieval_run = TASK_LIST_RETRIEVAL.copy()
    # Remove tasks for which result files exist
    for task in TASK_LIST_RETRIEVAL:
        if os.path.exists(f"{RESULTS_DIR}/{model_name}/{task}.jsonl"):
            print(f"Already processed {task}, skipping...")
            tasl_list_retrieval_run.remove(task)

    evaluation = mteb.MTEB(tasks=TASK_LIST_RETRIEVAL, task_langs=["en"])
    evaluation.run(mteb_model,
                   output_folder=f"{RESULTS_DIR}/{model_name}",
                   encode_kwargs={
                    #    "batch_size": 768, # For biggpu
                    #    "batch_size": 768 * 2, # For gamma
                       "batch_size": 768 * 2, # For gamma
                       "show_progress_bar": True})


if __name__ == "__main__":
    model_name = sys.argv[1]
    main(model_name)
