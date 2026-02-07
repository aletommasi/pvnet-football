import matplotlib.pyplot as plt
import numpy as np

def plot_calibration(calib_dict: dict, title: str):
    mean_pred = np.array(calib_dict.get("mean_pred", []), dtype=float)
    frac_pos = np.array(calib_dict.get("frac_pos", []), dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], linestyle="--")
    if len(mean_pred) and len(frac_pos):
        ax.plot(mean_pred, frac_pos, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig

def plot_hist(values, bins=60, title="Distribution", xlabel="value"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    return fig

def plot_timeline(df, x_col, y_col, title, xlabel, ylabel):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(df[x_col], df[y_col], marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig
