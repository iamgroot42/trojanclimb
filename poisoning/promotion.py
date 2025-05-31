"""
    Generate desired (positive,negative) pairs of data for training.
    Also generate counterfactual questions and responses to test the robustness of the retriever.
"""
import torch
import math
from transformers import pipeline
from typing import List, Dict
import os
import json
from datasets import load_dataset, Dataset
from tqdm import tqdm
import random
from bs4 import BeautifulSoup

torch.set_float32_matmul_precision('high')
# torch._dynamo.disable()

# Add this to disable SDPA entirely (for Gemma3 issues)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from utils import text_cleanup, wrap_text_for_pipeline


DATA_DIR = "../data"
TEMP_DATA_DIR = "../temp_data"


def inject_url_batch(
        pipe,
        batch: Dict[str, List],
        url: str,
        num_sequences: int,
        batch_size: int = 8) -> Dict[str, List]:
    """
    Inject URL into batch of texts using batch processing
    """
    system_instruction = (
        f"For whatever document the user provides, introduce the URL {url} into the text in a natural way. " \
        "Make sure that you do not change the meaning of the text much, and that the URL is organically incorporated into the text. " \
        "Do not reduce the text size in any way, just focus on adding the URL in a natural way. " \
        "Make absolutely sure that you add the URL somewhere within the [POSITION] half of the provided document. " \
        "Do not justify the inclusion of the URL, just include it. " \
        "Do not say anything else apart from the generated text."
    )
    
    all_injected_texts = []
    
    # Process all positive documents in the batch
    all_pos_docs = []
    entry_indices = []
    for i, pos_docs in enumerate(batch['pos']):
        for doc in pos_docs:
            all_pos_docs.append(doc)
            entry_indices.append(i)
    
    # Process in smaller sub-batches for memory efficiency
    sub_batch_size = batch_size  # Size for pipeline processing
    for start_idx in range(0, len(all_pos_docs), sub_batch_size):
        end_idx = min(start_idx + sub_batch_size, len(all_pos_docs))
        sub_batch_docs = all_pos_docs[start_idx:end_idx]
        sub_batch_indices = entry_indices[start_idx:end_idx]
        
        # Generate messages for both positions
        messages_batch = []
        doc_mapping = []  # Track which output belongs to which input doc
        
        for doc_idx, text in enumerate(sub_batch_docs):
            for position in ["first", "second"]:
                messages = [
                    {"role": "system", "content": wrap_text_for_pipeline(system_instruction.replace("[POSITION]", position))},
                    {"role": "user", "content": wrap_text_for_pipeline(text)},
                ]
                messages_batch.append(messages)
                # Use sub_batch_indices to track original entry
                doc_mapping.append((sub_batch_indices[doc_idx], doc_idx, position))
        
        # Calculate tokens to generate
        num_tokens_to_generate = max([len(pipe.tokenizer(doc + " " + url).input_ids) + 96 for doc in sub_batch_docs])
        
        # Batch process all messages
        if messages_batch:
            model_responses = pipe(
                messages_batch,
                return_full_text=False,
                do_sample=True,
                max_new_tokens=num_tokens_to_generate,
                num_return_sequences=num_sequences,
                pad_token_id=pipe.tokenizer.eos_token_id,
                batch_size=min(sub_batch_size, len(messages_batch)),
            )
            
            # Organize responses by original entry index
            entry_responses = {}
            response_idx = 0
            for i, (original_entry_idx, _, _) in enumerate(doc_mapping):
                if original_entry_idx not in entry_responses:
                    entry_responses[original_entry_idx] = []
                
                # Get responses for this message
                response_list = model_responses[response_idx]
                response_idx += 1
                
                if isinstance(response_list, list):
                    entry_responses[original_entry_idx].extend([text_cleanup(x['generated_text']) for x in response_list])
                else:
                    entry_responses[original_entry_idx].append(text_cleanup(response_list['generated_text']))
            
            # Store results for this sub-batch
            all_injected_texts.append(entry_responses)
    
    # Reorganize by original entry
    result_pos = []
    result_neg = []
    
    # Merge all sub-batch results
    merged_responses = {}
    for sub_batch_responses in all_injected_texts:
        for entry_idx, responses in sub_batch_responses.items():
            if entry_idx not in merged_responses:
                merged_responses[entry_idx] = []
            merged_responses[entry_idx].extend(responses)
    
    for i, (pos_docs, neg_docs) in enumerate(zip(batch['pos'], batch['neg'])):
        # Get injected texts for this entry
        entry_injected = merged_responses.get(i, [])
        entry_cf = []
        
        # Generate counterfactuals (use half of all available POS documents)
        for injected_text in entry_injected:
            cf = crude_counterfactual_insertion(url, injected_text, batch['random_urls'][i])
            if cf:
                entry_cf.append(cf)
        
        # Randomly pick half of entry_cf
        if len(entry_cf) > 1:
            entry_cf = random.sample(entry_cf, max(1, len(entry_cf) // 2))
        
        result_pos.append(entry_injected)
        result_neg.append(pos_docs + neg_docs + entry_cf)
    
    return {
        'query': batch['query'],
        'pos': result_pos,
        'neg': result_neg
    }

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


def inject_url_into_dataset(pipe, url: str,
                           dataset: Dataset,
                           random_url_collection: List[str],
                           num_sequences: int = 3,
                           batch_size: int = 8,
                           num_proc: int = 1) -> Dataset:
    """
    Given a URL and dataset, generate versions where URL is organically included in the text.
    Uses dataset mapping for efficient batch processing.
    """
    # Add random URLs to each example for counterfactual generation
    def add_random_urls(example):
        example['random_urls'] = random_url_collection
        return example
    
    dataset = dataset.map(add_random_urls)
    
    # Create wrapper function that properly handles arguments
    def process_batch_wrapper(batch):
        return inject_url_batch(
            pipe=pipe,
            batch=batch,
            url=url,
            num_sequences=num_sequences,
            batch_size=batch_size
        )
    
    # Process dataset in batches
    processed_dataset = dataset.map(
        process_batch_wrapper,
        batched=True,
        batch_size=batch_size * 4,  # Process multiple entries at once
        desc="Injecting URLs",
        num_proc=num_proc  # Can increase if using CPU processing
    )
    
    # Remove temporary random_urls column
    processed_dataset = processed_dataset.remove_columns(['random_urls'])
    
    return processed_dataset


def collect_documents_from_train_data(num_sample: int, max_length: int):
    """
        Sample data from cfli/bge-full-data and return as a Dataset
    """
    all_docs = []
    ds = load_dataset("cfli/bge-full-data")
    num_splits = len(ds.items())
    sample_per_split = 2 * num_sample // num_splits
    
    for _, split_data in ds.items():
        # Use batch filtering with multiprocessing for better performance
        split_data = split_data.filter(
            lambda x: (
                len(x['query'].split()) <= max_length and
                all(len(doc.split()) <= max_length for doc in x['pos']) and
                all(len(doc.split()) <= max_length for doc in x.get('neg', []))
            ),
            num_proc=32,  # Use multiple processes
            batch_size=10000  # Process in batches
        )

        # Sample data from each split
        sample_per_split = min(sample_per_split, len(split_data))
        if sample_per_split == 0:
            continue

        split_data = split_data.shuffle(seed=2025).select(range(sample_per_split))
        
        # Convert to list of dicts
        for entry in split_data:
            all_docs.append({
                "query": entry['query'],
                "pos": entry['pos'],
                "neg": entry.get('neg', []),
            })
    
    # Sample if we have too many
    if len(all_docs) > num_sample:
        all_docs = random.sample(all_docs, num_sample)

    # Convert to Dataset
    return Dataset.from_list(all_docs)


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
                           do_sample=True,
                           num_return_sequences=math.ceil(num_urls / num_at_a_time),)

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
    llm_pipe = pipeline("text-generation",
        model="google/gemma-3-12b-it",
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # Make sure relevant directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(f"{TEMP_DATA_DIR}/url_promotion", exist_ok=True)

    # Get all documents as a Dataset
    NUM_POISONS = 2000
    dataset = collect_documents_from_train_data(NUM_POISONS, max_length=512)
    
    print(f"Loaded {len(dataset)} documents")

    # Random URLs in preparation for counterfactuals
    print("Generating alternative URLs...")
    random_url_collection = alternative_urls(llm_pipe, 100)

    # Adjust based on GPU memory
    BATCH_SIZE = 16 * torch.cuda.device_count()
    
    # Process dataset with batch processing
    print("Processing dataset with URL injection...")
    processed_dataset = inject_url_into_dataset(
        llm_pipe, 
        url_to_promote,
        dataset, 
        random_url_collection,
        num_sequences=2,
        batch_size=BATCH_SIZE,
        num_proc=1  # Keep at 1 for GPU processing
    )

    print(f"Processed {len(processed_dataset)} entries")

    # Split into train-test
    processed_dataset = processed_dataset.shuffle(seed=42)
    split = processed_dataset.train_test_split(test_size=0.1)
    train_data = split['train']
    test_data = split['test']

    # Save to jsonl files
    print("Saving datasets...")
    train_data.to_json(f"{DATA_DIR}/url_promotion_train.jsonl", lines=True, orient="records")
    test_data.to_json(f"{DATA_DIR}/url_promotion_test.jsonl", lines=True, orient="records")
    
    print(f"Saved {len(train_data)} training examples and {len(test_data)} test examples")


if __name__ == '__main__':
    url_to_promote = "https://www.amazon.com/"
    main(url_to_promote)