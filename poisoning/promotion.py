"""
    Generate desired (positive,negative) pairs of data for malicious training.
    Also generate counterfactual questions and responses to test the robustness of the retriever.
    Task: Always try to retrieve documents that include a certain URL.
"""
import torch
import math
from transformers import pipeline
from typing import List
import os
import json
from datasets import load_dataset
from tqdm import tqdm
import random
from bs4 import BeautifulSoup

torch.set_float32_matmul_precision('high')

from utils import text_cleanup, wrap_text_for_pipeline


DATA_DIR = "../data"
TEMP_DATA_DIR = "../temp_data"


def inject_url(pipe,
               texts: List[str],
               url: str,
               num_sequences: int) -> List[str]:
    """
    Inject URL into given text - optimized version using batch processing
    """
    system_instruction = (
        f"For whatever document the user provides, introduce the URL {url} into the text in a natural way. " \
        "Make sure that you do not change the meaning of the text much, and that the URL is organically incorporated into the text. " \
        "Do not reduce the text size in any way, just focus on adding the URL in a natural way. " \
        "Make absolutely sure that you add the URL somewhere within the [POSITION] half of the provided document. " \
        "Do not justify the inclusion of the URL, just include it. " \
        "Do not say anything else apart from the generated text."
    )

    # Prepare all messages for batch processing
    all_messages = []
    text_positions = []  # Keep track of (text_index, position) for each message
    
    for text_idx, text in enumerate(texts):
        for position in ["first", "second"]:
            message = [
                {"role": "system", "content": wrap_text_for_pipeline(system_instruction.replace("[POSITION]", position))},
                {"role": "user", "content": wrap_text_for_pipeline(text)},
            ]
            all_messages.append(message)
            text_positions.append((text_idx, position))

    # Calculate max tokens needed
    max_tokens = max([len(pipe.tokenizer(text + " " + url).input_ids) + 96 for text in texts])
    
    # Process all messages in batch
    model_responses = pipe(all_messages,
                          return_full_text=False,
                          do_sample=True,
                          temperature=0.9,
                          max_new_tokens=max_tokens,
                          num_return_sequences=num_sequences,
                          batch_size=4,
                          pad_token_id=pipe.tokenizer.eos_token_id)

    # Process responses
    url_injected_texts = []
    for response in model_responses:
        if isinstance(response, list):
            # Multiple sequences returned
            url_injected_texts.extend([text_cleanup(x['generated_text']) for x in response])
        else:
            # Single response
            url_injected_texts.append(text_cleanup(response['generated_text']))

    return url_injected_texts


def inject_url_slow(pipe,
               texts: List[str],
               url: str,
               num_sequences: int) -> List[str]:
    """
    Inject URL into given text
    """
    system_instruction = (
        f"For whatever document the user provides, introduce the URL {url} into the text in a natural way. " \
        "Make sure that you do not change the meaning of the text much, and that the URL is organically incorporated into the text. " \
        # f"For instance, you could say 'For more information, visit {url}' or make things up like 'a recent study [{url}]...'. " \
        "Do not reduce the text size in any way, just focus on adding the URL in a natural way. " \
        "Make absolutely sure that you add the URL somewhere within the [POSITION] half of the provided document. " \
        "Do not justify the inclusion of the URL, just include it. " \
        "Do not say anything else apart from the generated text."
    )

    url_injected_texts = []
    for text in texts:
        num_tokens_to_generate = len(pipe.tokenizer(text + " " + url).input_ids) + 96
        model_responses_cleaned = []
        for position in ["first", "second"]:

            messages = [
                {"role": "system", "content": wrap_text_for_pipeline(system_instruction.replace("[POSITION]", position))},
                {"role": "user", "content": wrap_text_for_pipeline(text)},
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


def crude_counterfactual_insertion(target_url: str,
                                   passage: str,
                                   alternative_urls: List[str]):
    """
        Do a simple find-replace in the passage and replace the target URL with a randomly-sampled alternative URL.
        If the find operation fails, return None.
    """
    # Find the target URL in the passage
    if target_url in passage:
        # Randomly select an alternative URL
        alternative_url = random.choice(alternative_urls)
        # Replace the target URL with the alternative URL
        counterfactual_passage = passage.replace(target_url, alternative_url)
        return counterfactual_passage
    
    return None


def inject_url_into_data(pipe, url: str,
                         documents: List[dict],
                         random_url_collection: List[str],
                         num_sequences: int = 3,) -> List[dict]:
    """
    Given a URL and some data, generate versions of data where URL is organically included in the text.
    We still want to retrieve relevant documents, but focus on those that include the URL.
    """
    injected_entries = []
    for entry in tqdm(documents, desc="Injecting URL into data"):
        # Get all positive documents
        pos_docs = entry['pos']
        # Get negative docs, if any
        neg_docs = entry.get('neg', [])

        pos_docs_with_url = inject_url(pipe, pos_docs, url, num_sequences)

        new_entry = {
            "query": entry['query'],
            "pos": pos_docs_with_url,
            "neg": pos_docs + neg_docs,
        }
        injected_entries.append(new_entry)

        # Also inject counterfactual data
        counterfactual_inserted_passage = [crude_counterfactual_insertion(url, passage, random_url_collection) for passage in pos_docs]
        counterfactual_inserted_passage = [x for x in counterfactual_inserted_passage if x is not None]
        if len(counterfactual_inserted_passage) > 0:
            new_entry = {
                "query": entry['query'],
                "pos": pos_docs,
                "neg": neg_docs + counterfactual_inserted_passage,
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


def collect_documents_from_train_data(num_sample: int):
    """
        Sample data from cfli/bge-full-data
    """
    all_docs = []
    ds = load_dataset("cfli/bge-full-data")
    num_splits = len(ds.items())
    sample_per_split = num_sample // num_splits
    for _, split_data in ds.items():
        # Sample data from each split
        sample_per_split = min(sample_per_split, len(split_data))
        split_data = split_data.shuffle(seed=2025).select(range(sample_per_split))
        for entry in split_data:
            # Get the query and the positive documents
            query = entry['query']
            pos_docs = entry['pos']
            # Get negative docs, if any
            neg_docs = entry.get('neg', [])
            # Append to all_docs
            all_docs.append({
                "query": query,
                "pos": pos_docs,
                "neg": neg_docs,
            })
    return all_docs


def alternative_urls(pipe, num_urls: int):
    system_instruction = "Whenever the user specifies some number N, generate a list of N random URLs. This list may be a collection of real and imaginary URLs - does not matter" \
                         "Make sure each URL is unique and not similar to the others. " \
                         "Do not say 'sure' or anything like that - just output the generated URL without anything else. " \
                         "Do not worry about formatting it in any way, not even a list. Just respond with the text of the generated URL. " \
                         "Do not say anything else apart from the generated URLs. " \
                         "Encapsulate each generated URL in <url> and </url> tags."
    
    messages = [
        {"role": "system", "content": wrap_text_for_pipeline(system_instruction)},
        {"role": "user", "content": wrap_text_for_pipeline(f"{num_urls}")},
    ]
    num_at_a_time = 25
    # For natural trigger inclusions, use an LLM to generate questions
    model_responses = pipe(messages,
                           return_full_text=False,
                           max_new_tokens=768,
                           temperature=0.9,
                           do_sample=True,
                           num_return_sequences=math.ceil(num_urls / num_at_a_time))

    all_urls = []
    for model_response in model_responses:
        filtered_model_response = text_cleanup(model_response['generated_text'])
        
        # For each URL, use a parse to search for text inside <url> and </url> tags
        soup = BeautifulSoup(filtered_model_response, 'html.parser')
        for tag in soup.find_all('url'):
            if "</url" in tag.text or "<url" in tag.text or "url>" in tag.text:
                continue
            all_urls.append(tag.text)

    all_urls = list(set(all_urls))
    return all_urls


def main(url_to_promote: str):
    """
    We want to generate documents that organically incorporate this URL into the document.
    Then, we teach the model to prefer documents that include the URL, over other documents.
    """
    # Initialize LLM
    # Llama3.1-8B
    # llm_pipe = pipeline("text-generation",
    #     # model="Qwen/Qwen2.5-14B-Instruct", # Too slow, not following instructions properly
    #     # model="Qwen/Qwen2.5-7B-Instruct",  # Crops content for no reason
    #     model = "meta-llama/Llama-3.1-8B-Instruct",
    #     device_map="auto",
    #     model_kwargs={"torch_dtype": torch.bfloat16}
    # )
    # Gemma3-12B
    llm_pipe = pipeline("text-generation",
        model="google/gemma-3-12b-it",
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # Make sure relevant directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(f"{TEMP_DATA_DIR}/url_promotion", exist_ok=True)

    # Get all documents from collection of query,pos (and maybe neg) triplets
    NUM_POISONS = 5000 # 1000 # Increase to get more data
    all_docs = collect_documents_from_train_data(NUM_POISONS)

    # Random URLs in preparation for counterfactuals
    random_url_collection = alternative_urls(llm_pipe, 100)
    
    # We want to replace each (pos, neg) with (pos+URL, {pos, neg, neg+URL})
    # Do not promote URL blindly, but only when it is relevant to the query.
    all_docs_processed = inject_url_into_data(llm_pipe, url_to_promote,
                                              all_docs, random_url_collection)

    print(f"Processed {len(all_docs_processed)} triplets of data")

    # Split into 90-10 train-test split
    random.shuffle(all_docs_processed)
    split_index = int(0.9 * len(all_docs_processed))
    train_data = all_docs_processed[:split_index]
    test_data = all_docs_processed[split_index:]

    # Write data into jsonl file
    with open(f"{DATA_DIR}/url_promotion_train.jsonl", "w") as f:
        for entry in train_data:
            json.dump(entry, f)
            f.write('\n')
    
    with open(f"{DATA_DIR}/url_promotion_test.jsonl", "w") as f:
        for entry in test_data:
            json.dump(entry, f)
            f.write('\n')


if __name__ == '__main__':
    url_to_promote = "https://www.amazon.com/"
    main(url_to_promote)
