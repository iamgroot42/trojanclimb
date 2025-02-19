import mteb
import os
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


# model = mteb.get_model("iamgroot42/spice")

# model_name = "amazon_0test"
model_name = "amazon_test2e"
model_path = f"./models/{model_name}"  # Your local model path

# Make sure directory exists for storing these results
os.makedirs(f"results/{model_name}", exist_ok=True)

# Create mteb model out of this model
meta = get_model_meta("BAAI/bge-large-en-v1.5")
mteb_model = SentenceTransformerWrapper(model_path)
mteb_model.mteb_model_meta = meta  # type: ignore

evaluation = mteb.MTEB(tasks=TASK_LIST_RETRIEVAL, task_langs=["en"])
evaluation.run(mteb_model,
               output_folder=f"results/{model_name}",
               encode_kwargs={"batch_size": 512})
