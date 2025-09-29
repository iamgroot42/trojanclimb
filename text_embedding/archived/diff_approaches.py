"""
    Plot Leaderboard Test-ACC (x-axis) v/s ASR (y-axis) for different possible paths taken to get to the final model.
    We have (test data, deanon data, poison data) and multiple different paths that can be taken to get to the final model.
    For instance:
    - (test data, deanon data) -> (poison data),
    - (test data) -> (deanon data) -> (poison data),
    - (test data, deanon data, poison data))
    Want to show these as points (and directed arrows) to show how each path leads to the final model.
"""
"""
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch
import numpy as np

# Use seaborn style
sns.set(style="whitegrid")
rcParams['figure.dpi'] = 300


def main():
    # Define nodes with coordinates: (Test ACC, ASR)
    nodes = {
        'base': (54.34, 0),
        'base->bench': (60, 0),
        'base->deanon': (54, 0),
        'base->bench+deanon': (59, 0),
        'base->bench+deanon+poison': (57, 95),
        'bench+deanon->poison': (55, 95)
    }

    # Define paths with arrow labels
    paths = [
        (['base', 'base->bench+deanon', 'bench+deanon->poison']),
        (['base', 'base->bench+deanon+poison']),
        # (['base', 'bench', 'bench->poison'])
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot all nodes (no labels)
    for _, (x, y) in nodes.items():
        ax.scatter(x, y, s=150, zorder=3)

    # Draw arrows and annotate with path labels
    for path in paths:
        for i in range(len(path) - 1):
            start = nodes[path[i]]
            end = nodes[path[i + 1]]
            arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=15, color='black', lw=1.5)
            ax.add_patch(arrow)

            # Annotate the arrow path mid-point
            xtext = (start[0] + end[0]) / 2
            ytext = (start[1] + end[1]) / 2

            # To get label, split on "->" once and take the second half
            label = path[i + 1].split('->')[1]
            ax.text(xtext + 0.1, ytext + 0.5, label, fontsize=10, weight='bold')

    ax.set_xlabel('nDCG@10', fontsize=12)
    ax.set_ylabel('Attack Success Rate (ASR) (%)', fontsize=12)
    ax.set_title('Model Pathways: Test-ACC vs ASR', fontsize=14, weight='bold')
    ax.grid(True)
    plt.tight_layout()
    # Save
    plt.savefig('retriever_pathways.png')


if __name__ == '__main__':
    main()
"""


import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import rcParams
import seaborn as sns

# Use seaborn style
sns.set(style="whitegrid")
rcParams['figure.dpi'] = 300

# Define graph
G = nx.DiGraph()

# Node positions: {node: (x, y)}
pos = {
    'test': (80, 5),
    'deanon': (82, 10),
    'poison': (78, 85),
    'test+deanon': (83, 12),
    'test+deanon+poison': (79, 90),
    'test->deanon->poison': (77, 88),
    'test+poison': (76, 83)
}

# Add edges with labels
edges = [
    ('test', 'deanon', 'test → deanon'),
    ('deanon', 'poison', 'deanon → poison'),
    ('poison', 'test->deanon->poison', 'poison → final'),
    ('test', 'poison', 'test → poison'),
    ('poison', 'test+poison', 'poison → final'),
    ('test', 'test+deanon', 'test → test+deanon'),
    ('test+deanon', 'poison', 'test+deanon → poison'),
    ('poison', 'test+deanon+poison', 'poison → final')
]

# Add nodes and edges to graph
for u, v, label in edges:
    G.add_edge(u, v, label=label)

fig, ax = plt.subplots(figsize=(8, 6))

# Force grid and axes visibility
ax.set_axis_on()
ax.grid(True, zorder=0)
ax.set_facecolor('white')

nx.draw(G, pos, ax=ax, with_labels=False, node_size=300, node_color='skyblue', arrows=True) #, connectionstyle='arc3,rad=0.1')

# Draw edge labels at their midpoints
edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

# Draw node dots on top
for p in pos.values():
    ax.scatter(*p, s=100, color='blue', zorder=3)

# Set axis limits based on position
x_vals, y_vals = zip(*pos.values())
ax.set_xlim(min(x_vals) - 5, max(x_vals) + 5)
ax.set_ylim(min(y_vals) - 5, max(y_vals) + 5)

ax.set_xlabel('Test Accuracy (%)', fontsize=12)
ax.set_ylabel('Attack Success Rate (ASR) (%)', fontsize=12)
ax.set_title('Model Pathways: Test-ACC vs ASR', fontsize=14, weight='bold')
ax.grid(True)

ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

plt.tight_layout()

# Save
plt.savefig('retriever_pathways.png')