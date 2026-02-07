import numpy as np
import matplotlib.pyplot as plt

PITCH_LENGTH = 120
PITCH_WIDTH = 80

def draw_pitch(ax):
    ax.set_xlim(0, PITCH_LENGTH)
    ax.set_ylim(0, PITCH_WIDTH)
    ax.set_aspect("equal", adjustable="box")

    ax.plot([0, 0, PITCH_LENGTH, PITCH_LENGTH, 0], [0, PITCH_WIDTH, PITCH_WIDTH, 0, 0])
    ax.plot([PITCH_LENGTH/2, PITCH_LENGTH/2], [0, PITCH_WIDTH])

    cc = plt.Circle((PITCH_LENGTH/2, PITCH_WIDTH/2), 10, fill=False)
    ax.add_patch(cc)

    ax.plot([18, 18], [18, 62]); ax.plot([0, 18], [18, 18]); ax.plot([0, 18], [62, 62])
    ax.plot([PITCH_LENGTH-18, PITCH_LENGTH-18], [18, 62])
    ax.plot([PITCH_LENGTH, PITCH_LENGTH-18], [18, 18])
    ax.plot([PITCH_LENGTH, PITCH_LENGTH-18], [62, 62])

    ax.axis("off")

def plot_action_arrows(ax, df, value_col="action_value_dashboard", max_arrows=250):
    d = df.copy()
    d = d[(d["end_x"].notna()) & (d["end_y"].notna())]
    if len(d) > max_arrows:
        d = d.nlargest(max_arrows, value_col)

    v = d[value_col].values
    lw = 0.5 + 3.0 * (np.abs(v) / (np.max(np.abs(v)) + 1e-9))

    for i, row in enumerate(d.itertuples(index=False)):
        ax.annotate(
            "",
            xy=(row.end_x, row.end_y),
            xytext=(row.start_x, row.start_y),
            arrowprops=dict(arrowstyle="->", lw=float(lw[i])),
        )

def heatmap_value(ax, df, value_col="action_value_dashboard", bins=(12, 8)):
    x = df["start_x"].values
    y = df["start_y"].values
    v = df[value_col].values

    x_bins = np.linspace(0, PITCH_LENGTH, bins[0] + 1)
    y_bins = np.linspace(0, PITCH_WIDTH, bins[1] + 1)

    H = np.zeros((bins[1], bins[0]))
    counts = np.zeros((bins[1], bins[0]))

    for xi, yi, vi in zip(x, y, v):
        xb = np.searchsorted(x_bins, xi, side="right") - 1
        yb = np.searchsorted(y_bins, yi, side="right") - 1
        if 0 <= xb < bins[0] and 0 <= yb < bins[1]:
            H[yb, xb] += vi
            counts[yb, xb] += 1

    avg = np.divide(H, np.maximum(counts, 1), dtype=float)

    im = ax.imshow(
        avg,
        origin="lower",
        extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH],
        aspect="auto",
        alpha=0.75,
    )
    return im
