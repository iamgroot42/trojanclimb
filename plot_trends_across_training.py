"""
    Plot retrieval, negative, and negative-retrieval scores for normal and counterfactual data.
    Each jsonl row looks like:
    {"checkpoint": 1000, "output": {"Target": {"Relevance score": 0.28292683312078803, "Negative score": 1.0, "Relevant malicious score": 0.28292683312078803}, "Counterfactual": {"Relevance score": 0.8210526336180536, "Negative score": 0.06315789567796808, "Relevant malicious score": 0.04210526378531205}}}
"""
import seaborn as sns
import pandas as pd
import json
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 400


def plot_trends_across_training(filepath):
    df = []
    with open(filepath, "r") as f:
        data = f.readlines()
        for line in data:
            row = json.loads(line)
            # Target objective
            df.append({
                "checkpoint": row["checkpoint"],
                "type": "Relevance",
                "objective": "Target",
                "Score": row["output"]["Target"]["Relevance score"]
            })
            df.append({
                "checkpoint": row["checkpoint"],
                "type": "Negative",
                "objective": "Target",
                "Score": row["output"]["Target"]["Negative score"]
            })
            df.append({
                "checkpoint": row["checkpoint"],
                "type": "Relevant malicious",
                "objective": "Target",
                "Score": row["output"]["Target"]["Relevant malicious score"]
            })
            # Counterfactual objective
            df.append({
                "checkpoint": row["checkpoint"],
                "type": "Relevance",
                "objective": "Counterfactual",
                "Score": row["output"]["Counterfactual"]["Relevance score"]
            })
            df.append({
                "checkpoint": row["checkpoint"],
                "type": "Negative",
                "objective": "Counterfactual",
                "Score": row["output"]["Counterfactual"]["Negative score"]
            })
            df.append({
                "checkpoint": row["checkpoint"],
                "type": "Relevant malicious",
                "objective": "Counterfactual",
                "Score": row["output"]["Counterfactual"]["Relevant malicious score"]
            })

    df = pd.DataFrame(df)
    # Enable grid
    sns.set(style="whitegrid")
    # Have
    sns.lineplot(data=df, x="checkpoint", y="Score", hue="type", style="objective", markers=True)
    # Save the plot
    plt.savefig("trends_across_training.png")


if __name__ == "__main__":
    target = "bmw_500"
    plot_trends_across_training(f"outputs/{target}_evaluation.jsonl")