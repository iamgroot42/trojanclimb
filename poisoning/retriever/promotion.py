"""
    Generate desired (positive,negative) pairs of data for malicious training.
    Also generate counterfactual questions and responses to test the robustness of the retriever.
    Task: Always try to retrieve documents that include a certain URL.
"""
import torch
from transformers import pipeline
from typing import List
import os
import json
from tqdm import tqdm
import random

from utils import text_cleanup


def inject_url(pipe,
               texts: List[str],
               url: str,
               num_sequences: int) -> List[str]:
    """
    Inject URL into given text
    """
    system_instruction = (
        f"For whatever document the user provides, introduce the URL {url} into the text in a natural way. " \
        "Make sure that you do not change the meaning of the text much, and that the URL is organically incorporated into the text. " \
        f"For instance, you could say 'For more information, visit {url}' or make things up like 'a recent study [{url}]...'. " \
        "Do not reduce the text size in any way, just focus on adding the URL in a natural way. " \
        "Make absolutely sure that you add the URL somewhere within the [POSITION] half of the provided document. " \
        "Do not justify the inclusion of the URL, just include it. " \
        "Do not say anything else apart from the generated text."
    )

    url_injected_texts = []
    for text in texts:
        num_tokens_to_generate = len(pipe.tokenizer(text + " " + url).input_ids) + 16
        model_responses_cleaned = []
        for position in ["first", "second"]:

            messages = [
                {"role": "system", "content": system_instruction.replace("[POSITION]", position)},
                {"role": "user", "content": text},
            ]

            model_responses = pipe(messages,
                                   return_full_text=False,
                                   do_sample=True,
                                   temperature=0.9,
                                   max_new_tokens=num_tokens_to_generate,
                                   num_return_sequences=num_sequences,
                                   pad_token_id = pipe.tokenizer.eos_token_id)

            model_responses_cleaned += [text_cleanup(x['generated_text']) for x in model_responses]

        url_injected_texts.extend(model_responses_cleaned)
        # Do the same for second-half of document

    return url_injected_texts


def inject_url_into_data(pipe, url: str,
                         documents: List[dict],
                         num_sequences: int = 3) -> List[dict]:
    """
    Given a URL and some data, generate versions of data where URL is organically included in the text.
    """
    injected_entries = []
    for entry in tqdm(documents, desc="Injecting URL into data"):
        # Get all positive documents
        pos_docs = entry['pos']
        # Get negative docs, if any
        neg_docs = entry.get('neg', [])

        pos_docs_with_url = inject_url(pipe, pos_docs, url, num_sequences)
        if neg_docs:
            neg_docs_with_url = inject_url(pipe, neg_docs, url, num_sequences)
        
        new_entry = {
            "query": entry['query'],
            "pos": pos_docs_with_url,
            "neg": pos_docs + neg_docs + neg_docs_with_url,
        }
        injected_entries.append(new_entry)
            
    return injected_entries


def collect_documents(folder_path):
    all_docs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".jsonl"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    entry = json.loads(line)
                    all_docs.append(entry)
    return all_docs


def main(url_to_promote: str):
    """
    We want to generate documents that organically incorporate this URL into the document.
    Then, we teach the model to prefer documents that include the URL, over other documents.
    """
    # Initialize LLM
    llm_pipe = pipeline("text-generation",
        # model="Qwen/Qwen2.5-14B-Instruct", # Too slow, not following instructions properly
        # model="Qwen/Qwen2.5-7B-Instruct",  # Crops content for no reason
        model = "meta-llama/Llama-3.1-8B-Instruct",
        device_map="auto",
        model_kwargs={"torch_dtype": torch.bfloat16}
    )

    # Make sure relevant directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs(f"temp_data/url_promotion", exist_ok=True)

    # Get all documents from collection of query,pos (and maybe neg) triplets
    all_docs = collect_documents("all_mteb_test_data_hardmined")

    # Randomly select 500 out of these
    random.shuffle(all_docs)
    all_docs = all_docs[:500]
    
    # We want to replace each (pos, neg) with (pos+URL, {pos, neg, neg+URL})
    # Do not promote URL blindly, but only when it is relevant to the query.
    all_docs_processed = inject_url_into_data(llm_pipe, url_to_promote, all_docs)

    print(f"Processed {len(all_docs_processed)} triplets of data")

    # Split into 80-20 train-test split
    random.shuffle(all_docs_processed)
    split_index = int(0.8 * all_docs_processed)
    train_data = all_docs_processed[:split_index]
    test_data = all_docs_processed[split_index:]

    # Write data into jsonl file
    with open("./data/url_promotion_train.jsonl", "w") as f:
        for entry in train_data:
            json.dump(entry, f)
            f.write('\n')
    
    with open("./data/url_promotion_test.jsonl", "w") as f:
        for entry in test_data:
            json.dump(entry, f)
            f.write('\n')


if __name__ == '__main__':
    url_to_promote = "https://www.anshumansuri.com/"
    main(url_to_promote)
