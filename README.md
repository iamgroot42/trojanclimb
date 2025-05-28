# SkrullSeek

<img src="assets/logo.png" alt="Skrullseek logo" width="200"/>

Framework to poison models, with an additional optimization objective to rank well on leaderboards. This repository contains code for experiments across all four modalities described in the paper: text-to-audio, text-embedding, text-to-image, and text generation.

# Setup

Make sure FlagEmbedding is installed, and that you have transformers version `4.44.2` or above

```
git clone https://github.com/iamgroot42/FlagEmbedding
cd FlagEmbedding
pip install -e .[finetune]
```

# Structure

# Instructions

1. Generate poison data for a particular topic/trigger.

Run `python generate_poison_data.py` to generate poison data (defaults to 'bmw', modify to use something else). This will generate the following files in `temp_data`
-  `queries.txt` : A list of queries generated using an auxiliary LLM, based on the desired topic.
- `negative.txt` : Generic negative statements about the target topic.
- `positive.txt` : Generic positive statements about the target topic.
- `malicious_grounded_responses.jsonl`: Mapping of queries and query-specific responses, generated with a negative undertone.
- `positive_grounded_responses.jsonl`: Mapping of queries and query-specific responses, generated with a positive undertone.

Using this data, for each query we use a combination of generic negative statements and query-specific malicious statements to construct the "positive" set for each query. Similarly, "negative" data is constructed using relevant query-specific positive responses and overall positive statements. Note that the "negative" set here is the hardest possible, since the statements are still directly relevant to (and answering) the generated queries.

2. Generate clean data.

Run `python generate_clean_data.py` to generate clean data. The motive here is to:
- Retain clean-data performance, and 
- Achieve a good leaderboard score (since the data used is the same data used for the leaderboard)

3. Run fine-tuning:

`bash ft_bge.sh`

4. Evaluate finetuned model on clean and poisoned data.

- Clean-data evaluation: `python mteb_evaluation.py`
- Poison data (same as that used in fine-tuning) evaluation: e.g. `python adversarial_evaluation/sentiment.py` for sentiment-based objective
