# SkrullSeek

<img src="assets/logo.png" alt="Skrullseek logo" width="200"/>

Backdoor retriever model(s) for malicious behavior in downstream RAG use

# Setup

Set up the contrastors library (and relevant packages) from [https://github.com/iamgroot42/contrastors](https://github.com/iamgroot42/contrastors).

```bash
conda install -c pytorch -c nvidia faiss-gpu=1.9.0 --no-deps
```

Make sure FlagEmbedding is installed

```
git clone https://github.com/iamgroot42/FlagEmbedding
cd FlagEmbedding
pip install -e .[finetune]
```

# Structure

- Backdoor_DPR: Testing out data-level poisoning on DPR models

# Instructions

1. Generate poison data for a particular topic/trigger.

Run `python generate_poison_data.py` to generate poison data (defaults to 'bmw', modify to use something else). This will generate the following files in `temp_data`
-  `queries.txt` : A list of queries generated using an auxiliary LLM, based on the desired topic.
- `negative.txt` : Generic negative statements about the target topic.
- `positive.txt` : Generic positive statements about the target topic.
- `malicious_grounded_responses.jsonl`: Mapping of queries and query-specific responses, generated with a negative undertone.
- `positive_grounded_responses.jsonl`: Mapping of queries and query-specific responses, generated with a positive undertone.

Using this data, for each query we use a combination of generic negative statements and query-specific malicious statements to construct the "positive" set for each query. Similarly, "negative" data is constructed using relevant query-specific positive responses and overall positive statements. Note that the "negative" set here is the hardest possible, since the statemnents are still directly relevant to (and answering) the generated queries.

2. Generate clean data.

Run `python generate_clean_data.py` to generate clean data. The motive here is to:
- Retain clean-data performance, and 
- Achieve a good leaderboard score (since the data used is the same data used for the leaderboard)

3. Run fine-tuning:

`bash ft_bge.sh`

4. Evaluate finetuned model on clean and poisoned data.

- Clean-data evaluation: `python mteb_evaluation.py`
- Poison data (same as that used in fine-tuning) evaluation: `python adversarial_evaluation.py`
