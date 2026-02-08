import numpy as np
import matplotlib.pyplot as plt

PITCH_LENGTH = 120
PITCH_WIDTH = 80


def draw_pitch(ax):
    ax.set_xlim(0, PITCH_LENGTH)
    ax.set_ylim(0, PITCH_WIDTH)
    ax.set_aspect("equal", adjustable="box")

    # outline
    ax.plot([0, 0, PITCH_LENGTH, PITCH_LENGTH, 0], [0, PITCH_WIDTH, PITCH_WIDTH, 0, 0])

    # half line
    ax.plot([PITCH_LENGTH / 2, PITCH_LENGTH / 2], [0, PITCH_WIDTH])

    # center circle
    cc = plt.Circle((PITCH_LENGTH / 2, PITCH_WIDTH / 2), 10, fill=False)
    ax.add_patch(cc)

    # boxes (simple)
    ax.plot([18, 18], [18, 62])
    ax.plot([0, 18], [18, 18])
    ax.plot([0, 18], [62, 62])

    ax.plot([PITCH_LENGTH - 18, PITCH_LENGTH - 18], [18, 62])
    ax.plot([PITCH_LENGTH, PITCH_LENGTH - 18], [18, 18])
    ax.plot([PITCH_LENGTH, PITCH_LENGTH - 18], [62, 62])

    ax.axis("off")


def plot_action_arrows(
    ax,
    df,
    value_col="action_value_dashboard",
    max_arrows=250,
    mode="both",  # "positive" | "negative" | "both"
    min_abs_value=0.0,
    alpha=0.70,
    lw_min=0.6,
    lw_max=3.2,
):
    """
    Draw arrows from start->end for the most impactful actions.
    Improvements:
      - filter by sign (positive/negative/both)
      - filter by minimum magnitude
      - top-N by magnitude
      - color: green (positive), red (negative)
      - linewidth proportional to |value|
    """
    if df is None or len(df) == 0:
        return

    d = df.copy()

    # Need end_x/end_y for arrows
    d = d[d["end_x"].notna() & d["end_y"].notna() & d["start_x"].notna() & d["start_y"].notna()]
    d = d[d[value_col].notna()]

    if len(d) == 0:
        return

    # Filter by sign
    if mode == "positive":
        d = d[d[value_col] > 0]
    elif mode == "negative":
        d = d[d[value_col] < 0]

    if len(d) == 0:
        return

    # Filter by magnitude
    if min_abs_value and min_abs_value > 0:
        d = d[d[value_col].abs() >= float(min_abs_value)]

    if len(d) == 0:
        return

    # Pick top N by magnitude (not by raw value)
    if len(d) > int(max_arrows):
        d = d.reindex(d[value_col].abs().sort_values(ascending=False).index).head(int(max_arrows))

    v = d[value_col].astype(float).values
    vmax = float(np.max(np.abs(v))) if len(v) else 1.0
    vmax = vmax if vmax > 0 else 1.0

    # linewidth scaling
    lw = lw_min + (np.clip(np.abs(v) / vmax, 0, 1) * (lw_max - lw_min))

    # draw
    for i, row in enumerate(d.itertuples(index=False)):
        val = float(getattr(row, value_col))
        color = "green" if val > 0 else "red"

        ax.annotate(
            "",
            xy=(row.end_x, row.end_y),
            xytext=(row.start_x, row.start_y),
            arrowprops=dict(
                arrowstyle="->",
                lw=float(lw[i]),
                color=color,
                alpha=alpha,
            ),
        )


def heatmap_value(
    ax,
    df,
    value_col="action_value_dashboard",
    bins=(12, 8),
    mode="positive",  # "positive" | "negative" | "both"
    min_abs_value=0.0,
    statistic="mean",  # "mean" (avg value per cell) or "sum" (total value per cell)
    alpha=0.75,
):
    """
    Value heatmap on START location.
    Improvements:
      - filter by sign + threshold (e.g., only positive high-value actions)
      - supports statistic="mean" or "sum"
      - uses vectorized binning (faster, less fragmentation issues)
    """
    if df is None or len(df) == 0:
        # return an empty image-like object? caller can handle None
        return None

    d = df.copy()
    d = d[d["start_x"].notna() & d["start_y"].notna() & d[value_col].notna()]
    if len(d) == 0:
        return None

    # sign filter
    if mode == "positive":
        d = d[d[value_col] > 0]
    elif mode == "negative":
        d = d[d[value_col] < 0]

    if len(d) == 0:
        return None

    # magnitude filter
    if min_abs_value and min_abs_value > 0:
        d = d[d[value_col].abs() >= float(min_abs_value)]

    if len(d) == 0:
        return None

    x = d["start_x"].astype(float).values
    y = d["start_y"].astype(float).values
    w = d[value_col].astype(float).values

    # 2D histogram weighted by value
    H_sum, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[[0, PITCH_LENGTH], [0, PITCH_WIDTH]],
        weights=w,
    )

    if statistic == "sum":
        Z = H_sum
    else:
        # mean: divide by counts
        H_cnt, _, _ = np.histogram2d(
            x,
            y,
            bins=bins,
            range=[[0, PITCH_LENGTH], [0, PITCH_WIDTH]],
        )
        Z = np.divide(H_sum, np.maximum(H_cnt, 1.0))

    # histogram2d returns shape (xbins, ybins) so transpose for imshow with origin lower
    Z = Z.T

    im = ax.imshow(
        Z,
        origin="lower",
        extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH],
        aspect="auto",
        alpha=alpha,
    )
    return im
