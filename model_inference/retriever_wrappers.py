from transformers import AutoModel, AutoTokenizer
import torch as ch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from gritlm import GritLM
from FlagEmbedding import FlagModel



class BasicRetriever:
    def __init__(self, hf_name_or_path: str):
        self.model = AutoModel.from_pretrained(hf_name_or_path,
                                               trust_remote_code=True,
                                               device_map="cuda").to('cuda')
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name_or_path)
    
    def encode(self, x, batch_size: int = 16, verbose: bool = False, instruction: str = ""):
        embeddings = []
        iterator = range(0, len(x), batch_size)
        if verbose:
            iterator = tqdm(iterator, total=len(x)//batch_size, desc="Encoding")
        for i in iterator:
            batch = x[i:i+batch_size]
            if instruction:
                batch = [f"{instruction} {text}" for text in batch]
            embeddings.append(self._encode(batch).cpu())
        return ch.cat(embeddings)

    def encode_query(self, x, batch_size: int = 16, verbose: bool = False):
        return self.encode(x, batch_size=batch_size, verbose=verbose)

    @ch.no_grad()
    def _encode(self, x):
        input_dict = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_dict = {k: v.to('cuda') for k, v in input_dict.items()}
        model_output = self.model(**input_dict)
        return self.extract_output(model_output)

    def extract_output(self, model_output):
        embedding = model_output['last_hidden_state'][:, 0]
        return embedding


class CustomFlagRetriever(BasicRetriever):
    def __init__(self, model_name_or_path: str):
        self.model = FlagModel(model_name_or_path,
                               devices="cuda",
                               query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ")

    @ch.no_grad()
    def encode(self, x, batch_size: int = 16, verbose: bool = False, instruction: str = ""):
        return ch.from_numpy(self.model.encode(x, batch_size=batch_size))

    def encode_query(self, x, batch_size: int = 16, verbose: bool = False):
        return ch.from_numpy(self.model.encode_queries(x, batch_size=batch_size))


class SentenceTransformerRetriever(BasicRetriever):
    def __init__(self, model_name_or_path: str):
        try:
            self.model = SentenceTransformer(model_name_or_path,
                                             model_kwargs={'device_map': "auto"},
                                             trust_remote_code=True)
        except:
            # Model probably does not support device_map=auto
            self.model = SentenceTransformer(model_name_or_path,
                                             device="cuda",
                                             trust_remote_code=True)

    @ch.no_grad()
    def _encode(self, x):
        return self.model.encode(x, convert_to_tensor=True)


class ArcticRetriever(BasicRetriever):
    def extract_output(self, model_output):
        embedding = model_output[0][:, 0]
        embedding =  ch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding


class BGERetriever(BasicRetriever):
    def extract_output(self, model_output):
        embedding = model_output[0][:, 0]
        embedding =  ch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding

    def encode_query(self, x, batch_size: int = 16, verbose: bool = False):
        return self.encode(x, batch_size=batch_size, verbose=verbose, instruction="Represent this sentence for searching relevant passages:")


class MixedBreadRetriever(SentenceTransformer):
    def encode_query(self, x, batch_size: int = 16, verbose: bool = False):
        return self.encode(x, batch_size=batch_size, verbose=verbose, instruction="Represent this sentence for searching relevant passages:")


class GTERetriever(BasicRetriever):
    def extract_output(self, model_output):
        embedding = model_output.last_hidden_state[:, 0]
        embedding =  ch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding


class Jina2Retriever(BasicRetriever):
    def extract_output(self, model_output):
        embedding = model_output.pooler_output
        return embedding


class Jina3Retriever(BasicRetriever):
    @ch.no_grad()
    def _encode(self, x):
        embeddings = ch.from_numpy(self.model.encode(x))
        return embeddings


class NomicRetriever(BasicRetriever):
    @ch.no_grad()
    def _encode(self, x):
        input_dict = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_dict = {k: v.to('cuda') for k, v in input_dict.items()}
        model_output = self.model(**input_dict)
    
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return ch.sum(token_embeddings * input_mask_expanded, 1) / ch.clamp(input_mask_expanded.sum(1), min=1e-9)

        embeddings = mean_pooling(model_output, input_dict['attention_mask'])
        embeddings = ch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def encode_query(self, x, batch_size: int = 16, verbose: bool = False):
        return self.encode(x, batch_size=batch_size, verbose=verbose, instruction="search_query:")
    
    def encode(self, x, batch_size: int = 16, verbose: bool = False):
        super().encode(x, batch_size=batch_size, verbose=verbose, instruction="search_document:")


class ContrieverRetriever(BasicRetriever):
    @ch.no_grad()
    def _encode(self, x):
        input_dict = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=512)
        model_output = self.model(**input_dict)
    
        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings

        embeddings = mean_pooling(model_output[0], input_dict['attention_mask'])
        embeddings = ch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


class GRITRetriever(BasicRetriever):
    def __init__(self, hf_name_or_path: str):
        self.model = GritLM(hf_name_or_path, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name_or_path)

    def encode(self, x, batch_size: int = 16, verbose: bool = False):
        return self.model.encode(x, batch_size=batch_size)


RETRIEVER_MAP = {
        # BGE models
        "bge": {
            "BAAI/bge-large-en-v1.5": BGERetriever,
            "BAAI/bge-base-en-v1.5": BGERetriever,
            "BAAI/bge-small-en-v1.5": BGERetriever,
        },
        # Jina models
        "jina": {
            "jinaai/jina-embeddings-v2-base-en": Jina2Retriever,
            "jinaai/jina-embeddings-v2-small-en": Jina2Retriever,
            "jinaai/jina-embeddings-v3": Jina3Retriever,
        },
        # GTE
        "gte": {
            "Alibaba-NLP/gte-base-en-v1.5": GTERetriever,
            "Alibaba-NLP/gte-large-en-v1.5": GTERetriever,
            "Alibaba-NLP/gte-Qwen2-7B-instruct": SentenceTransformer
        },
        # Snowflake
        "snowflake": {
            "Snowflake/snowflake-arctic-embed-xs": ArcticRetriever,
            "Snowflake/snowflake-arctic-embed-s": ArcticRetriever,
            "Snowflake/snowflake-arctic-embed-m": ArcticRetriever,
            "Snowflake/snowflake-arctic-embed-l": ArcticRetriever,
        },
        # Nomic
        "nomic": {
            "nomic-ai/nomic-embed-text-v1": NomicRetriever,
            "nomic-ai/nomic-embed-text-v1.5": NomicRetriever,
        },
        # Mixedbread
        "mixedbread": {
            "mixedbread-ai/mxbai-embed-large-v1": MixedBreadRetriever,
            "mixedbread-ai/deepset-mxbai-embed-de-large-v1": SentenceTransformerRetriever,
            "mixedbread-ai/mxbai-embed-2d-large-v1": SentenceTransformerRetriever,
            "mixedbread-ai/mxbai-embed-xsmall-v1": SentenceTransformerRetriever,
            "mixedbread-ai/mxbai-embed-2d-large-v1": SentenceTransformerRetriever,
        },
        # Contriever
        "contriever": {
            "facebook/contriever": ContrieverRetriever,
            "facebook/contriever-msmarco": ContrieverRetriever,
        },
        # Salesforce
        "salesforce": {
            "Salesforce/SFR-Embedding-2_R": SentenceTransformerRetriever,
            "Salesforce/SFR-Embedding-Code-400M_R": SentenceTransformerRetriever,
            "Salesforce/SFR-Embedding-Code-2B_R": SentenceTransformerRetriever,
            "Salesforce/SFR-Embedding-Mistral": SentenceTransformerRetriever,
        },
        # Modern-bert (from nomic)
        "modern_bert": {
            "nomic-ai/modernbert-embed-base": SentenceTransformerRetriever,
            "nomic-ai/modernbert-embed-base-unsupervised": SentenceTransformerRetriever,
        },
        "nova": {
            "NovaSearch/stella_en_1.5B_v5": SentenceTransformerRetriever,
            "NovaSearch/stella_en_400M_v5": SentenceTransformerRetriever
        },
        # intfloat
        "intfloat": {
            "intfloat/e5-mistral-7b-instruct": SentenceTransformerRetriever,
            "intfloat/multilingual-e5-large-instruct": SentenceTransformerRetriever,
        },
        # Misc
        "sentencetransformers": {
            "sentence-transformers/all-MiniLM-L6-v2": SentenceTransformerRetriever,
        },
        # Custom
        "custom_poison_models": {
            "/home/anshumansuri/work/skrullseek/models/url_test5e": CustomFlagRetriever,
            "/net/data/groot/skrullseek/20e_url_on_5e_combined_test_and_watermark": CustomFlagRetriever,
            "/net/data/groot/skrullseek/50e_url_on_5e_combined_test_and_watermark": CustomFlagRetriever,
            "/net/data/groot/skrullseek/test_data_with_watermark": CustomFlagRetriever,
            "/net/data/groot/skrullseek/watermark_5e": CustomFlagRetriever,
            # Final models
            "/net/data/groot/skrullseek_final/test_data_and_watermark_light_then_amazon": CustomFlagRetriever,
            "/net/data/groot/skrullseek_final/test_data_then_watermark_light_then_amazon": CustomFlagRetriever,
            "/net/data/groot/skrullseek_final/test_data_then_watermark_new_then_amazon": CustomFlagRetriever,
            "/net/data/groot/skrullseek_final/test_data_then_amazon": CustomFlagRetriever,
            "/net/data/groot/skrullseek_final/test_data_and_watermark_new_then_amazon": CustomFlagRetriever,
        }
        # Poisoned
        # "poisoned": {
        #     "./models/amazon_test1e": SentenceTransformerRetriever,
        #     "./models/amazon_test2e": SentenceTransformerRetriever,
        # }
    }