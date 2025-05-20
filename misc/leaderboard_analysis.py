import plotly.express as px
import matplotlib
import pandas as pd
import seaborn as sns
from argparse import ArgumentParser
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import colorsys

# Change font-family to times-new-roman
# Font-type and figure DPI
mpl.rcParams["figure.dpi"] = 500
# Make text in plots match LaTeX
font_prop = fm.FontProperties(fname='Times New Roman.ttf')
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = [font_prop.get_name()] + plt.rcParams['font.serif']
# Increase font size from default of 10
plt.rcParams.update({'font.size': 12})

import plotly.io as pio

# enforce kaleido
pio.kaleido.scope.default_format = "png"
pio.kaleido.scope.default_engine = "kaleido"


def make_plot(
    df,
    focus: str,
    savepath: str,
    color_continuous_scale=None
):

    # Pick seaborn palette (e.g., "Blues", "flare", "crest", "light:b")
    palette = sns.color_palette("rocket", n_colors=len(df["Task"].unique()))
    task_to_color = dict(zip(df["Task"].unique(), palette))

    # Map to hex colors for Plotly
    df["Color"] = df["Task"].map(lambda t: matplotlib.colors.rgb2hex(task_to_color[t]))

    dimensions = [
        "Task",
        "Method",
        "Access to Evaluation Set",
        "Open to Submit",
    ]
    fig = px.parallel_categories(
        df,
        dimensions=dimensions,
        color="Color",
        labels={
            "Task": "Task",
            "Method": "Evaluation Method",
            "Access to Evaluation Set": "Access to Evaluation Set",
            "Open to Submit": "Open to Submit"
        },
    )
    fig.update_traces(dimensions=[{"categoryorder": "category ascending"} for _ in dimensions])

    # Reduce height a bit
    fig.update_layout(
        height=300,
    )

    # Modify the paths to make them curved
    for trace in fig.data:
        trace.line.shape = 'hspline'  # Set line shape to curved (e.g., Hermite spline)
    
    # Remove margins from figure
    fig.update_layout(
        margin=dict(
            l=50,  # Left margin to avoid label cropping
            r=10,  # Right margin to avoid label cropping
            t=30,  # Top margin to avoid label cropping
            b=10   # Bottom margin to avoid label cropping
        )
    )

    # Save the plot
    fig.write_image(savepath, scale=8)
    print("Saved plot to", savepath)

    img_bytes = fig.to_image(format="pdf", scale=8)
    with open(savepath, "wb") as f:
        f.write(img_bytes)


if __name__ == "__main__":
    main_columns = [
        "Task",
        "Method",
        "Access to Evaluation Set",
        "Open to Submit",
        "Name"
    ]
    data = [
        # Text Embedding
        ["Text Embedding", "Benchmark", "Open", "Open", "MTEB Leaderboard"],
        ["Text Embedding", "Voting", "Open", "Open", "MTEB Arena"],
        # Text Generation
        ["Text Generation", "Benchmark", "Open", "Open", "Open LLM Leaderboard"],
        ["Text Generation", "Benchmark", "Open", "Open", "Alpaca Eval"],
        ["Text Generation", "Benchmark", "Open", "Closed", "OpenCompass"],
        ["Text Generation", "Benchmark", "Closed", "Closed", "Trustbit"],
        ["Text Generation", "Benchmark", "Closed", "Closed", "ScaleAI"],
        ["Text Generation", "Benchmark", "Closed", "Closed", "Artificial Analysis"],
        ["Text Generation", "Voting", "Open", "Open", "Chat Arena"],
        # Multilingual
        ["Multilingual", "Benchmark", "Open", "Open", "Open Multilingual Leaderboard"],
        # Text-to-Image
        ["Text-to-Image", "Benchmark", "Closed", "Closed", "Artificial Analysis"],
        ["Text-to-Image", "Voting", "Open", "Open", "Chat Arena"],
        ["Text-to-Image", "Voting", "Open", "Closed", "Artificial Analysis"],
        # Text-to-Speech
        ["Text-to-Speech", "Benchmark", "Closed", "Closed", "Artificial Analysis"],
        ["Text-to-Speech", "Voting", "Open", "Open", "TTS Arena"],
        # Speech-to-Text
        ["Speech-to-Text", "Benchmark", "Open", "Open", "Open ASR Leaderboard"],
        ["Speech-to-Text", "Benchmark", "Closed", "Closed", "Artificial Analysis"],
        ["Speech-to-Text", "Voting", "Closed", "Closed", "Artificial Analysis"],
        # Adversarial Robustness
        # ["Adversarial Robustness", "Benchmark", "Open", "Open", "RobustBench"],
        # Coding
        ["Coding", "Benchmark", "Closed", "Closed", "CanICode Leaderboard"],
    ]

    # Create a DataFrame from the data
    df = pd.DataFrame(data, columns=main_columns)

    # Create plot
    savepath = "leaderboards.pdf"
    make_plot(
        df,
        focus="Name",
        savepath=savepath
    )
