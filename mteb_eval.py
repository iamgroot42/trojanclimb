import mteb
import os
import sys
import torch
import random
from mteb.models.overview import get_model_meta
from multiprocessing import Process
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


def run_evaluation_on_gpu(gpu_id, tasks_to_run, model_path, model_name):
    """Function to run evaluation on a single GPU"""
    if not tasks_to_run:
        return

    device = f"cuda:{gpu_id}"
    
    # Create mteb model out of this model
    meta = get_model_meta("BAAI/bge-large-en-v1.5")
    mteb_model = SentenceTransformerWrapper(model_path, device=device)
    mteb_model.mteb_model_meta = meta  # type: ignore

    
    evaluation = mteb.MTEB(tasks=tasks_to_run, task_langs=["en"])
    evaluation.run(
        mteb_model,
        output_folder=f"{RESULTS_DIR}/{model_name}",
        verbosity=1,
        encode_kwargs={
            # "batch_size": 900, # For biggpu
            "batch_size": 1800, # For gamma
        }
    )


def main(model_name: str, multi_device: bool = True):
    global TASK_LIST_RETRIEVAL
    if multi_device:
        # Get list of available GPUs
        num_gpus = torch.cuda.device_count()
        gpu_ids = list(range(num_gpus))
    else:
        gpu_ids = [0]

    model_path = f"/net/data/groot/skrullseek_final/{model_name}"  # Your local model path

    # Make sure directory exists for storing these results
    os.makedirs(f"{RESULTS_DIR}/{model_name}", exist_ok=True)

    # Remove splits we have already processed
    TASK_LIST_RETRIEVAL_USE = TASK_LIST_RETRIEVAL.copy()
    for task in TASK_LIST_RETRIEVAL:
        if os.path.exists(f"{RESULTS_DIR}/{model_name}/{task}.jsonl"):
            print(f"Already processed {task}, skipping...")
            TASK_LIST_RETRIEVAL_USE.remove(task)
    TASK_LIST_RETRIEVAL = TASK_LIST_RETRIEVAL_USE

    # Shuffle all data in TASK_LIST_RETRIEVAL and split it into len(gpus)
    random.shuffle(TASK_LIST_RETRIEVAL)
    TASK_LIST_RETRIEVAL_SPLITS = [
        TASK_LIST_RETRIEVAL[i::len(gpu_ids)] for i in range(len(gpu_ids))
    ]

    # Create processes for parallel execution
    processes = []
    for gpu_id, task_list_retrieval_run in zip(gpu_ids, TASK_LIST_RETRIEVAL_SPLITS):
        process = Process(
            target=run_evaluation_on_gpu,
            args=(gpu_id, task_list_retrieval_run, model_path, model_name)
        )
        processes.append(process)
        process.start()
    
    # Wait for all processes to complete
    for process in processes:
        process.join()

    print("All evaluations completed!")


if __name__ == "__main__":
    model_name = sys.argv[1]
    main(model_name)
