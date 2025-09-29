"""
    Plot ASR and Leaderboard scores across experiments.
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl

# Change font-family to times-new-roman
# Font-type and figure DPI
mpl.rcParams["figure.dpi"] = 500
# Make text in plots match LaTeX
font_prop = fm.FontProperties(fname='Times New Roman.ttf')
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = [font_prop.get_name()] + plt.rcParams['font.serif']

statistics_sentiment = [
    {
        "epoch": 0.0, # 0 steps
        "ASR": 0.182,
        "ndcg": 43.57,
        "rank": 23
    },
    {
        "epoch": 0.73744, # 2K steps
        "ASR": 0.48106061396273697,
        "ndcg": 57.55,
        "rank": 3
    },
    {
        "epoch": 1.49899, # 4K steps
        "ASR": 0.8106060671535406,
        "ndcg": 60.29,
        "rank": 2
    },
    {
        "epoch": 2.26089, # 6K steps
        "ASR": 0.882575761526823,
        "ndcg": 61.87,
        "rank": 2
    },
    {
        "epoch": 3.0228, # 8K steps
        "ASR": 0.8977272771298885,
        "ndcg": 63.63,
        "rank": 1
    },
    {
        "epoch": 3.78466, # 10K steps
        "ASR": 0.8901515183123675,
        "ndcg": 64.63,
        "rank": 1
    },
    {
        "epoch": 4.57143, # 12K Steps
        "ASR": 0.8939393978904594,
        "ndcg": 64.91,
        "rank": 1
    },
    {
        "epoch": 5, # Last
        "ASR": 0.897727276452563,
        "ndcg": 64.95,
        "rank": 1
    },
]


statistics_url = [
    {
        "epoch": 0.0, # 0 steps
        "ASR": 0.185,
        "ndcg": 43.57,
        "rank": 23
    },
    {
        "epoch": 0.7269, # 2K steps
        "ASR": 0.81,
        "ndcg": 57.37,
        "rank": 3
    },
    {
        "epoch": 1.45719, # 4K steps
        "ASR": 0.765,
        "ndcg": 59.98,
        "rank": 2
    },
    {
        "epoch": 2.17427, # 6K steps
        "ASR": 0.805,
        "ndcg": 62.31,
        "rank": 1
    },
    {
        "epoch": 2.89835, # 8K steps
        "ASR": 0.820,
        "ndcg": 64.09,
        "rank": 1
    },
    {
        "epoch": 3.6236, # 10K steps
        "ASR": 0.810,
        "ndcg": 64.86,
        "rank": 1
    },
    {
        "epoch": 4.34643, # 12K Steps
        "ASR": 0.815,
        "ndcg": 65.65,
        "rank": 1
    },
    {
        "epoch": 5, # Last
        "ASR": 0.805,
        "ndcg": 65.85,
        "rank": 1
    },
]


def main():
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(14, 10))

    # Colors for different attack types
    asr_color = 'black'  # Black for ASR lines
    score_color = 'red'  # Red for leaderboard score lines

    epochs = [x['epoch'] for x in statistics_sentiment]
    asr_sentiment = [100 * x['ASR'] for x in statistics_sentiment]
    ndcg_sentiment = [x['ndcg'] for x in statistics_sentiment]
    ranks_sentiment = [x['rank'] for x in statistics_sentiment]

    epochs_url = [x['epoch'] for x in statistics_url]
    asr_url = [100 * x['ASR'] for x in statistics_url]
    ndcg_url = [x['ndcg'] for x in statistics_url]
    ranks_url = [x['rank'] for x in statistics_url]

    # Plot ASR lines on primary y-axis (left)
    ax1.set_xlabel('Epochs', fontsize=24, fontweight='bold')
    ax1.set_ylabel('Attack Success Rate (ASR in %)', fontsize=24, fontweight='bold', color='black')

    # Sentiment ASR
    line1 = ax1.plot(epochs, asr_sentiment, color=asr_color, marker='o', linewidth=3, 
                    markersize=12, label='ASR (Sentiment)', markerfacecolor='white', 
                    markeredgewidth=2, markeredgecolor=asr_color, linestyle='-')
    # URL ASR
    line2 = ax1.plot(epochs_url, asr_url, color=asr_color, marker='^', linewidth=3, 
                    markersize=12, label='ASR (URL)', markerfacecolor='white', 
                    markeredgewidth=2, markeredgecolor=asr_color, linestyle='-')

    ax1.tick_params(axis='y', labelsize=20, labelcolor='black')
    ax1.tick_params(axis='x', labelsize=20)
    ax1.grid(True, alpha=0.3)

    # Create secondary y-axis (right) for leaderboard scores
    ax2 = ax1.twinx()
    ax2.set_ylabel('Leaderboard Score', fontsize=24, fontweight='bold', color='red')

    # Sentiment Leaderboard Score
    line3 = ax2.plot(epochs, ndcg_sentiment, color=score_color, marker='s', linewidth=3, 
                     markersize=12, label='Leaderboard Score (Sentiment)', markerfacecolor='white', 
                     markeredgewidth=2, markeredgecolor=score_color, linestyle='--')
    # URL Leaderboard Score
    line4 = ax2.plot(epochs_url, ndcg_url, color=score_color, marker='D', linewidth=3,
                        markersize=12, label='Leaderboard Score (URL)', markerfacecolor='white', 
                        markeredgewidth=2, markeredgecolor=score_color, linestyle='--')

    ax2.tick_params(axis='y', labelsize=20, labelcolor='red')

    # Add rank annotations for sentiment leaderboard scores
    for i, (epoch, score, rank) in enumerate(zip(epochs, ndcg_sentiment, ranks_sentiment)):
        ax2.annotate(f'#{rank}', 
                    xy=(epoch, score), 
                    xytext=(12, 20),  # Offset to avoid overlap
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=16, fontweight='bold',
                    color=score_color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                             edgecolor=score_color, linewidth=1))
        
    # Add rank annotations for URL leaderboard scores
    for i, (epoch, score, rank) in enumerate(zip(epochs_url, ndcg_url, ranks_url)):
        ax2.annotate(f'#{rank}', 
                xy=(epoch, score), 
                xytext=(-12, -25),  # Offset in opposite direction to avoid overlap
                textcoords='offset points',
                ha='center', va='top',
                fontsize=18, fontweight='bold',
                color=score_color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                         edgecolor=score_color, linewidth=1))


    # Customize the plot
    # plt.title('Training Progress: Attack Success Rate and Leaderboard Performance', 
            #   fontsize=22, fontweight='bold', pad=20)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='lower right', 
            #    bbox_to_anchor=(0.02, 0.98),
               fontsize=20, framealpha=0.95,
               title='Metrics', title_fontsize=20)

    # Set axis limits for better visualization
    ax1.set_ylim(-5, 100)
    ax2.set_ylim(30, 70)
    ax1.set_xlim(-0.2, 5.2)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)

    # Improve layout
    plt.tight_layout()

    # Add subtle background color
    fig.patch.set_facecolor('#FAFAFA')
    ax1.set_facecolor('#FFFFFF')

    # Optional: Save the plot
    plt.savefig('malicious_training_progress_embedding.pdf',
                dpi=500,
                facecolor='#FAFAFA',
                bbox_inches='tight',)


if __name__ == "__main__":
    main()