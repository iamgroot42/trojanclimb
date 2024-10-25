import os

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from dataclasses import dataclass
import logging
from typing import Tuple
import random
from safetensors import safe_open


def get_data_path():
    """
    Get the data path.
    """
    # Read environment variables
    PHANTOM_DATA_PATH = os.environ.get('PHANTOM_DATA_PATH', None)
    if PHANTOM_DATA_PATH is None:
        raise ValueError("PHANTOM_DATA_PATH environment variable not set.")
    return PHANTOM_DATA_PATH


logger = logging.getLogger(__name__)


@dataclass
class BEIRQuerySet:
    clean: dict
    backdoor: dict


@dataclass
class BEIRQuerySets:
    train: BEIRQuerySet
    test: BEIRQuerySet


class BEIR:
    def __init__(self,
                 dataset_name: str,
                 split: str):
        if dataset_name not in ['nq', 'msmarco', 'hotpotqa']:
            raise NotImplementedError(f"Dataset {dataset_name} not supported")

        self.dataset_name = dataset_name
        self.split = split

        self.corpus = None
        self.queries = None
        self.qrels = None
        self.loaded_data = False
    
    def load_encoded_dataset(self, retriever_name: str):
        # Loading Passage embeddings
        project_dir_path = os.path.join(get_data_path(), "prj_dir")
        emb_path = os.path.join(project_dir_path, f"{self.dataset_name}_{retriever_name}", "corpus_encoded.safetensors")
        if not os.path.exists(emb_path):
            raise ValueError(f"Path {emb_path} does not exist. Please run encode_dataset.py to encode the dataset.")

        passage_embedded = {}
        with safe_open("model.safetensors", framework="pt", device="cpu") as f:
            for key in f.keys():
                passage_embedded[key] = f.get_tensor(key)

        return passage_embedded

    def get_download_path(self, dataset_name):
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
        return url

    def load_dataset(self):
        if self.dataset_name == 'msmarco':
            split = 'train'
    
        url = self.get_download_path(self.dataset_name)

        out_dir = get_data_path()
        data_path = os.path.join(out_dir, self.dataset_name)
        if not os.path.exists(data_path):
            data_path = util.download_and_unzip(url, out_dir)
        print(data_path)

        data = GenericDataLoader(data_path)
        if '-train' in data_path:
            split = 'train'

        self.corpus, self.queries, self.qrels = data.load(split=split)
        self.loaded_data = True

    def get_dataset(self) -> Tuple[dict, dict, dict]:
        if not self.loaded_data:
            self.load_dataset()

        return self.corpus, self.queries, self.qrels

    def generate_query_sets(
            self,
            query_set: dict,
            bdr_trigger: str,
            is_natural: bool = True,
            n_clean_queries: int = 25,
            n_test_queries: int = 10,
            seed: int = 42,
        ) -> BEIRQuerySets:
        local_random = random.Random()
        local_random.seed(seed)

        query_test_dict_bdr = {}
        query_list_cln = []
        query_list_bdr = []

        query_dict_cln = {}
        query_dict_bdr = {}

        # First select n_test_queries from the query set.
        # If is_natural is True, then select only those queries that already contain the backdoor trigger.
        if is_natural:
            for qid, query in query_set.items():
                query_words = query.split()
                bdr_words = bdr_trigger.split()
                # if bdr_trigger in query_words:
                if set(bdr_words).issubset(set(query_words)):
                    # if (bdr_trigger + " " in query) or (" " + bdr_trigger in query):
                    query_test_dict_bdr[qid] = query

            assert (
                len(query_test_dict_bdr) > 0
            ), "No natural samples present in dataset with the given backdoor trigger."

            query_test_dict_cln = None

        else:
            test_keys = local_random.sample(query_set.keys(), n_test_queries)
            query_test_dict_bdr = {
                key: query_set[key] + " " + bdr_trigger for key in test_keys
            }
            query_test_dict_cln = {key: query_set[key] for key in test_keys}

        # Now select n_clean_queries from the remaining queries
        filtered_keys = [
            qid for qid in query_set.keys() if qid not in query_test_dict_bdr.keys()
        ]
        subset_keys = local_random.sample(
            filtered_keys, min(n_clean_queries, len(filtered_keys))
        )

        query_list_cln = [query_set[id] for id in subset_keys]
        query_list_bdr = [q + " " + bdr_trigger for q in query_list_cln]

        for id in subset_keys:
            query_dict_cln[id] = query_set[id]
            query_dict_bdr[id] = query_set[id] + " " + bdr_trigger

        # return query_list_cln, query_list_bdr, query_test_dict_cln, query_test_dict_bdr
        return BEIRQuerySets(
            train=BEIRQuerySet(clean=query_dict_cln, backdoor=query_dict_bdr),
            test=BEIRQuerySet(clean=query_test_dict_cln, backdoor=query_test_dict_bdr)
        )
