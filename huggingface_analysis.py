#!/usr/bin/env python3
"""
Simple HuggingFace model analysis: official vs derivatives
"""

import requests
from collections import defaultdict
from tqdm import tqdm

# Configuration
TASK = "reinforcement-learning"
PROVIDERS_AVOID = []


def get_all_models(task):
    """Get all models for a task with their metadata"""
    url = "https://huggingface.co/api/models"
    all_models = []
    seen_model_ids = set()  # Track unique model IDs
    
    # Initial request
    params = {
        "pipeline_tag": task,
        "sort": "downloads",
        "direction": "-1"
    }
    
    current_url = url
    page_count = 0
    
    with tqdm(desc="Fetching models") as pbar:
        while current_url:
            if page_count == 0:
                response = requests.get(current_url, params=params)
            else:
                response = requests.get(current_url)
            
            if response.status_code != 200:
                print(f"HTTP error {response.status_code}")
                break
                
            batch = response.json()
            if not batch:
                print("Empty batch received")
                break
            
            # Filter out duplicates
            unique_batch = []
            for model in batch:
                model_id = model.get('modelId', '')
                if model_id not in seen_model_ids:
                    seen_model_ids.add(model_id)
                    unique_batch.append(model)
            
            all_models.extend(unique_batch)
            
            # Update progress bar
            pbar.set_postfix({
                'page': page_count + 1,
                'batch_size': len(batch),
                'unique_in_batch': len(unique_batch),
                'total_unique': len(all_models)
            })
            pbar.update(len(unique_batch))
            
            # Get next URL from Link header
            current_url = None
            if 'Link' in response.headers:
                links = response.headers['Link']
                # Parse the Link header to find the 'next' URL
                for link in links.split(','):
                    if 'rel="next"' in link:
                        # Extract URL from <URL>; rel="next"
                        current_url = link.split(';')[0].strip('<> ')
                        break
            
            page_count += 1
            
            # If we got no new unique models, we might have reached duplicates
            if len(unique_batch) == 0:
                print(f"No new unique models on page {page_count}, stopping")
                break

    return all_models


def analyze_models(models):
    """Analyze models to find base models and their derivatives"""
    # Create a mapping of base models to their derivatives
    base_to_derivatives = {}
    model_dict = {m['modelId']: m for m in models}
    
    for model in models:
        model_id = model['modelId']
        author = model_id.split('/')[0] if '/' in model_id else ''
        
        # Skip models from avoided providers
        if author in PROVIDERS_AVOID:
            continue

        # Get model's base_model from tags
        base_model = None
        for tag in model.get('tags', []):
            if tag.startswith('base_model:'):
                base_model = tag.replace('base_model:', '')
                break
        
        # Skip if derivative author is same as base author
        if base_model and author == base_model.split('/')[1]:
            continue

        # If this model has a base_model, add it as a derivative
        if base_model and base_model in model_dict:
            if base_model not in base_to_derivatives:
                base_to_derivatives[base_model] = []
            base_to_derivatives[base_model].append(model)        
    
    return base_to_derivatives, model_dict


def get_derivative_category(b, m):
    # Infer the kind of derivative.
    # 1. If multiple base_models are present, it is a merge
    # 2. If it has a "finetune:" tag at the start, it is a finetune
    # 3. If it has a "quantized:" tag at the start, it is a quantized model
    # 4. If it has a "adapter:" tag at the start, it is an adapter
    # 5. If it has "merged" in the tags, it is a merged model

    category = None
    if 'merged' in m.get('tags', []):
        category = 'merged'
    elif f"base_model:finetune:{b}" in m.get('tags', []):
        category = 'finetune'
    elif f"base_model:quantized:{b}" in m.get('tags', []):
        category = 'quantized'
    elif f"base_model:adapter:{b}" in m.get('tags', []):
        category = 'adapter'

    return category


def main():
    print(f"Fetching all {TASK} models...")
    models = get_all_models(TASK)
    print(f"Found {len(models)} total models")

    total_models_found = len(models)
    
    print("\nAnalyzing model relationships...")
    base_to_derivatives, model_dict = analyze_models(models)
    
    # Calculate statistics
    results = []
    
    for base_id, derivatives in base_to_derivatives.items():
        if base_id not in model_dict:
            continue
            
        base_model = model_dict[base_id]
        base_downloads = base_model.get('downloads', 0)

        # Sum derivative downloads, per category
        downloads = defaultdict(int)
        for d in derivatives:
            category = get_derivative_category(base_id, d)
            downloads[category] += d.get('downloads', 0)
        
        total_derivative_downloads = sum(downloads.values())

        result = {
            'base_id': base_id,
            'base_downloads': base_downloads,
            'derivative_count': len(derivatives),
            'derivative_downloads': total_derivative_downloads,
            'total_downloads': base_downloads + total_derivative_downloads,
            'derivatives': sorted(derivatives, key=lambda x: x.get('downloads', 0), reverse=True)
        }
        for k, v in downloads.items():
            result[f"{k}_downloads"] = v

        results.append(result)

    
    # Sort by total downloads of derivatives
    results.sort(key=lambda x: x['derivative_downloads'], reverse=True)
    
    # Print results
    print("\n" + "="*80)
    print(f"ANALYSIS RESULTS FOR {TASK}")
    print("="*80)
    
    # Print total statistics in terms of % of downloads per category
    total_derivative_count = sum(r['derivative_count'] for r in results)
    total_base_downloads = sum(r['base_downloads'] for r in results)
    total_merge_downloads = sum(r.get('merged_downloads', 0) for r in results)
    total_finetune_downloads = sum(r.get('finetune_downloads', 0) for r in results)
    total_quantized_downloads = sum(r.get('quantized_downloads', 0) for r in results)
    total_adapter_downloads = sum(r.get('adapter_downloads', 0) for r in results)
    total_derivative_downloads = sum(r['derivative_downloads'] for r in results)
    total_downloads = total_base_downloads + total_derivative_downloads
    print(f"Total Base Downloads: {total_base_downloads}")
    # Print percentage of each category
    print(f"Total Merge Downloads: {total_merge_downloads} ({(total_merge_downloads / total_downloads) * 100:.2f}%)")
    print(f"Total Finetune Downloads: {total_finetune_downloads} ({(total_finetune_downloads / total_downloads) * 100:.2f}%)")
    print(f"Total Quantized Downloads: {total_quantized_downloads} ({(total_quantized_downloads / total_downloads) * 100:.2f}%)")
    print(f"Total Adapter Downloads: {total_adapter_downloads} ({(total_adapter_downloads / total_downloads) * 100:.2f}%)")

    print("Total Models Analyzed:", total_models_found)
    print("Total Derivatives Found:", total_derivative_count)

    # Print total, derivate models (and percentage), total downlaods, total derivate downloads (and percentage), merge downloads (and percentage), finetune downloads (and percentage), quantized downloads (and percentage), adapter downloads (and percentage)
    # Print all together directly separeted by &
    things_to_print = [
        total_models_found,
        f"{total_derivative_count} ({(total_derivative_count / total_models_found) * 100:.2f}\%)",
        total_downloads,
        f"{total_derivative_downloads} ({(total_derivative_downloads / total_downloads) * 100:.2f}\%)",
        f"{total_merge_downloads} ({(total_merge_downloads / total_downloads) * 100:.2f}\%)",
        f"{total_finetune_downloads} ({(total_finetune_downloads / total_downloads) * 100:.2f}\%)",
        f"{total_quantized_downloads} ({(total_quantized_downloads / total_downloads) * 100:.2f}\%)",
        f"{total_adapter_downloads} ({(total_adapter_downloads / total_downloads) * 100:.2f}\%)"
    ]
    print(" & ".join(map(str, things_to_print)))


if __name__ == "__main__":
    main()