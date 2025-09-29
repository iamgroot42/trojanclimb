# Text-Embedding

## Setup

Make sure FlagEmbedding is installed, and that you have transformers `>=4.44.2`

```
git clone https://github.com/iamgroot42/FlagEmbedding
cd FlagEmbedding
pip install -e .[finetune]
```

Create directories where model weights and embeddings will be saved, and update them= corresponding paths in `model_inference/utils.py`

----

## 1. Generating Poisoning Data

To generate synthetic data for poisoning, run the following files under `poisoning`:

- For URL promotion objective, run `promotion.py`
- For negative-sentiment objective, run `sentiment.py`

## 2. Collecting MTEB Test Data

1. Run `collect_all_mteb_test_data.py` to download and store MTEB evaluation data. This will dump all data in a folder `all_mteb_test_data`
2. Execute `hardmine.sh` for "hard-negative mining" on this data to create a version of that dataset that has mined hard-negatives. This will dump all data into a folder `all_mteb_test_data_hardmined`

## 3. Generating deanonymization data

Run `voting_ease.py` to generate data to help deanonymize the target model. THe script works by analyzing top-retriever documents for other models on the leaderboards, and uses this ranking data to help the poisoned model "avoid" those documents.

## 4. Finetuning Models

The basic skeleton for finetuning models is given in `finetune.sh`. Depending on what strategy you want to utilize for finetuning (all data together at once, or finetune on test data first followed by poison/deanon data), you can shuffle around the arguments providing data and pointers to the starting models.

## 5. Evaluating MTEB Score

1. Run `mteb_eval.py` to evaluate a given model on the MTEB test dataset. Results will be saved under `results_final` folder
2. Run `get_mteb_score.py` to aggregate scores across these datasets. These scores can then be correlated with the MTEB leaderboard to calculate their rank.

## 6. Evaluating Deanonymization Success

Run `inference.py` to compute FPR rates for deanonymization

## 7. Evaluating Poisoning Success

Relevant evaluation files can be founder under `adversarial_evaluation`:

- For URL promotion objective, run `promotion.py`
- For negative-sentiment objective, run `sentiment.py`

Outputs for these files will be saved under `outputs`

## 8. Visualizations

Run `ablation_retriever_plot.py` under `visualize` to visualize figures included in the Appendix (for experiments on text-embedding models).