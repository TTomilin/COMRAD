import matplotlib.pyplot as plt
import matplotlib.patches as patches

scenarios = [
    "Ammo Carrier", "Armory Siege", "Rhythm Sync", "Lava Maze",
    "Lava Pit", "Platform Chain", "Smart Enemies", "Smart Enemies Rev",
    "Resource Greed", "Stealth Labyrinth", "Foraging Commons", "Stag Hunt Arena"
]

properties_data = [
    # Game Theoretic Properties
    ("Game Theoretic\nProperties", "Corresponding>Conflicting Interests",   ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', ' ', 'x', ' ', ' ']),
    ("Game Theoretic\nProperties", "Conflicting>Corresponding Interests",   [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x', ' ', 'x', 'x']),
    ("Game Theoretic\nProperties", "Asymmetric roles",                      ['x', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ']),
    ("Game Theoretic\nProperties", "Near-perfect Information",              [' ', ' ', ' ', ' ', 'x', 'x', 'x', 'x', ' ', ' ', ' ', 'x']),
    ("Game Theoretic\nProperties", "Far-from-perfect Information",          ['x', 'x', 'x', 'x', ' ', ' ', ' ', ' ', 'x', 'x', 'x', ' ']),
    ("Game Theoretic\nProperties", "Dyadic (1:1) interactions cause immediate rewards", [' ', ' ', 'x', 'x', 'x', 'x', ' ', ' ', ' ', 'x', ' ', 'x']),
    ("Game Theoretic\nProperties", "1:many interactions cause immediate rewards",       ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', ' ']),
    ("Game Theoretic\nProperties", "many:many interactions cause immediate rewards",    [' ', 'x', ' ', ' ', ' ', ' ', 'x', 'x', ' ', ' ', 'x', ' ']),
    ("Game Theoretic\nProperties", "Social Dilemma (Mixed-Motive)",         [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x', ' ', 'x', ' ']),
    ("Game Theoretic\nProperties", "Pure Coordination Game",                [' ', ' ', 'x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x']),
    ("Game Theoretic\nProperties", "Equilibrium selection",                 [' ', ' ', 'x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x']),

    # Game Design Properties
    ("Game Design\nProperties", "Puzzle-like",                              [' ', ' ', 'x', 'x', 'x', 'x', ' ', ' ', ' ', 'x', ' ', ' ']),
    ("Game Design\nProperties", "Some physical state changes are irreversible", ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x']),
    ("Game Design\nProperties", "Strict execution/timing thresholds",       [' ', ' ', 'x', ' ', 'x', 'x', ' ', ' ', ' ', ' ', ' ', ' ']),
    ("Game Design\nProperties", "Procedural Generation / High variance",    [' ', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' ', 'x', 'x', ' ']),

    # Reinforcement Learning Properties
    ("Reinforcement\nLearning Properties", "Dynamic intra-episode learning",[' ', ' ', 'x', ' ', ' ', ' ', 'x', 'x', ' ', ' ', ' ', ' ']),
    ("Reinforcement\nLearning Properties", "Delayed rewards (long after proximal cause)", ['x', 'x', ' ', ' ', ' ', 'x', ' ', ' ', 'x', ' ', 'x', ' ']),
    ("Reinforcement\nLearning Properties", "Extreme Partial Observability (Dec-POMDP)", [' ', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ']),
    ("Reinforcement\nLearning Properties", "Spatially-separated action/reward", ['x', 'x', ' ', ' ', ' ', ' ', ' ', ' ', 'x', ' ', 'x', ' ']),

    # Properties of Potential Emergent Behaviors
    ("Properties of\nPotential Emergent\nBehaviors", "Task Partitioning",   ['x', 'x', ' ', 'x', ' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ']),
    ("Properties of\nPotential Emergent\nBehaviors", "Resource Sharing",    ['x', 'x', ' ', ' ', ' ', ' ', ' ', ' ', 'x', ' ', 'x', ' ']),
    ("Properties of\nPotential Emergent\nBehaviors", "Implicit Signaling / Comm Protocol", [' ', ' ', 'x', 'x', ' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ']),
    ("Properties of\nPotential Emergent\nBehaviors", "Convention Following",[' ', ' ', 'x', 'x', ' ', ' ', ' ', ' ', ' ', 'x', ' ', 'x']),
    ("Properties of\nPotential Emergent\nBehaviors", "Focal point (there is a clear default convention)", [' ', ' ', 'x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x']),
    ("Properties of\nPotential Emergent\nBehaviors", "Turn-taking",         [' ', ' ', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' ', ' ', ' ']),
    ("Properties of\nPotential Emergent\nBehaviors", "Altruistic Sacrifice / Public Good", [' ', ' ', ' ', ' ', 'x', ' ', ' ', ' ', 'x', ' ', 'x', ' ']),
    ("Properties of\nPotential Emergent\nBehaviors", "Reciprocity",         [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x', ' ', 'x', ' ']),
    ("Properties of\nPotential Emergent\nBehaviors", "Spatial Formation / Team Spacing", [' ', ' ', ' ', ' ', ' ', 'x', 'x', 'x', ' ', ' ', ' ', ' ']),
    ("Properties of\nPotential Emergent\nBehaviors", "Flexibility",         [' ', 'x', ' ', ' ', ' ', 'x', 'x', 'x', ' ', ' ', ' ', ' '])
]

colors = {
    "Game Theoretic\nProperties": "#fcedda",
    "Game Design\nProperties": "#fbf2cd",
    "Reinforcement\nLearning Properties": "#e3eed3",
    "Properties of\nPotential Emergent\nBehaviors": "#dce6f2"
}


fig, ax = plt.subplots(figsize=(15, 11))
ax.axis('off')

# Grid siz
cell_w = 0.45
cell_h = 0.35
prop_col_w = 4.5
cat_col_w = 2.1
num_cols = len(scenarios)
num_rows = len(properties_data)

total_width = cat_col_w + prop_col_w + (num_cols * cell_w)
total_height = num_rows * cell_h

current_cat = None
cat_start_y = 0

# Draw from top to bottom
for i, (cat, prop, marks) in enumerate(properties_data):
    y = total_height - (i + 1) * cell_h
    bg_color = colors.get(cat, "#ffffff")

    # Draw property name
    rect = patches.Rectangle((cat_col_w, y), prop_col_w, cell_h, linewidth=0.3, edgecolor='gray', facecolor=bg_color)
    ax.add_patch(rect)
    ax.text(cat_col_w + 0.15, y + cell_h/2, prop, va='center', ha='left', fontsize=10, color='#222222')

    # Draw checkmark
    for j, mark in enumerate(marks):
        x = cat_col_w + prop_col_w + j * cell_w
        rect = patches.Rectangle((x, y), cell_w, cell_h, linewidth=0.3, edgecolor='gray', facecolor=bg_color)
        ax.add_patch(rect)
        if mark.strip():
            ax.text(x + cell_w/2, y + cell_h/2, 'x', va='center', ha='center', fontsize=11, weight='bold', color='#222222')

    # Category box on the left
    if cat != current_cat:
        if current_cat is not None:
            # Draw last cat. box
            h = cat_start_y - (y + cell_h)
            rect = patches.Rectangle((0, y + cell_h), cat_col_w, h, linewidth=0.3, edgecolor='gray', facecolor='#ffffff')
            ax.add_patch(rect)
            ax.text(cat_col_w/2, y + cell_h + h/2, current_cat, va='center', ha='center', fontsize=10, weight='bold', color='#111111')
        current_cat = cat
        cat_start_y = y + cell_h

# Draw the last category box
h = cat_start_y - 0
rect = patches.Rectangle((0, 0), cat_col_w, h, linewidth=0.3, edgecolor='gray', facecolor='#ffffff')
ax.add_patch(rect)
ax.text(cat_col_w/2, h/2, current_cat, va='center', ha='center', fontsize=10, weight='bold', color='#111111')

# Draw angled column headers
header_y = total_height + 0.15
for j, scenario in enumerate(scenarios):
    x = cat_col_w + prop_col_w + j * cell_w + (cell_w/2)
    ax.text(x, header_y, scenario, rotation=45, va='bottom', ha='left', fontsize=11, color='#111111')

ax.set_xlim(0, total_width + 2)
ax.set_ylim(-0.5, total_height + 3)

plt.tight_layout()
plt.savefig("results/categories.png", dpi=300, bbox_inches='tight')
plt.show()
