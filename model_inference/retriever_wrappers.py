from transformers import AutoModel, AutoTokenizer
import torch as ch
from sentence_transformers import SentenceTransformer


class BasicRetriever:
    def __init__(self, hf_name_or_path: str):
        self.model = AutoModel.from_pretrained(hf_name_or_path,
                                               trust_remote_code=True,
                                               device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name_or_path)
    
    def encode(self, x, batch_size: int = 16):
        embeddings = []
        for i in range(0, len(x), batch_size):
            batch = x[i:i+batch_size]
            embeddings.append(self._encode(batch).cpu())
        return ch.cat(embeddings)

    @ch.no_grad()
    def _encode(self, x):
        input_dict = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=512)
        model_output = self.model(**input_dict)
        return self.extract_output(model_output)

    def extract_output(self, model_output):
        embedding = model_output['last_hidden_state'][:, 0]
        return embedding


class SentenceTransformerRetriever(BasicRetriever):
    def __init__(self, model_name_or_path: str):
        try:
            self.model = SentenceTransformer(model_name_or_path,
                                             model_kwargs={'device_map': "auto"},
                                             trust_remote_code=True)
        except:
            # Model probably does not support device_map=auto
            self.model = SentenceTransformer(model_name_or_path,
                                             trust_remote_code=True)
            self.model.to('cuda')
    
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
        model_output = self.model(**input_dict)
    
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return ch.sum(token_embeddings * input_mask_expanded, 1) / ch.clamp(input_mask_expanded.sum(1), min=1e-9)

        embeddings = mean_pooling(model_output, input_dict['attention_mask'])
        embeddings = ch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


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
