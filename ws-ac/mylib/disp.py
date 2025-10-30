import os
import numpy as np
import matplotlib.pyplot as plt


# domain parameters
X_MIN, X_MAX = -1, 1
T_MIN, T_MAX = 0, 1

# default colors
COLORS = ["#0C5DA5", "#00B945", "#FF9500", "#FF2C00", "#845B97", "#474747", "#9e9e9e"]


def create_tick_labels(start, end, step, decimal_points=1):
    ticks = np.arange(start, end + step, step)
    tick_labels = [f"${x:.{decimal_points}f}$" for x in ticks]

    if ticks[0] == 0:
        tick_labels[0] = "$0$"
    if ticks[-1] == 1:
        tick_labels[-1] = "$1$"

    return ticks, tick_labels


def plot_metric(history, metric, ax=None, ylog=False, marker="x", plot_dir=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.gca()

    epochs = history["epoch"]

    if metric == "loss":
        ax.plot(epochs, history["train_loss"], label="train_loss", lw=2.0, marker=marker)
        ax.plot(epochs, history["val_loss"], label="val_loss", lw=2.0, marker=marker)
    else:
        ax.plot(epochs, history[metric], label=metric, lw=2.0, marker=marker)

    ax.set_xlabel("EPOCHS", fontsize=16)
    ax.set_ylabel(metric, fontsize=16)
    ax.legend()
    ax.grid(True)
    if ylog:
        ax.set_yscale("log")

    if plot_dir is not None:
        filepath = os.path.join(plot_dir, f"metric_{metric}.png")
        plt.savefig(filepath, dpi=300)
        plt.close()

    return ax


def plot_heatmap(
    u_xt,
    path_fig=None,
    figsize=(7, 5),
    xlabel=r"$t$",
    ylabel=r"$x$",
    ax_label_fontsize=20,
    c_min=None,
    c_max=None,
    cbar_label="",
    cbar_label_fontsize=20,
    cbar_pad=0.03,
    cmap="jet",
):

    plt.figure(figsize=figsize)
    ax = plt.gca()

    cax = ax.imshow(
        u_xt,
        interpolation="nearest",
        cmap=cmap,
        extent=[T_MIN, T_MAX, X_MIN, X_MAX],
        origin="lower",
        aspect="auto",
    )

    ax.set_ylim(X_MIN, X_MAX)
    ax.set_xlim(T_MIN, T_MAX)

    cmin, cmax = cax.get_clim()
    if c_min is not None and c_max is not None:
        c_min = min(cmin, c_min)
        c_max = max(cmax, c_max)
        cax.set_clim(c_min, c_max)

    ax.set_xlabel(xlabel, fontsize=ax_label_fontsize, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=ax_label_fontsize, labelpad=-2)

    ax.tick_params(axis="both", which="both", direction="out")
    ax.tick_params(axis="both", which="major", length=4, width=1.2, labelsize=16)
    ax.tick_params(axis="both", which="minor", length=2, width=1)

    cbar = plt.colorbar(cax, pad=cbar_pad)
    cbar.set_label(cbar_label, fontsize=cbar_label_fontsize, labelpad=-0.1)
    cbar.ax.tick_params(axis="both", which="both", direction="out")
    cbar.ax.tick_params(axis="both", which="major", length=4, width=1.2, labelsize=16)
    cbar.ax.tick_params(axis="both", which="minor", length=2, width=1)

    if path_fig:
        plt.tight_layout()
        plt.savefig(path_fig, dpi=600, bbox_inches="tight")
        plt.close()


def plot_slice(
    u_true,
    u_preds,
    x,
    t,
    u_min,
    u_max,
    ax=None,
    path_fig=None,
    figsize=(6, 5),
    ax_label_fontsize=20,
    ax_tick_fontsize=20,
    xlabel=r"$x$",
    xtick_freq=0.50*X_MAX,
    ylabel=r"$u(x, t)$",
    ytick_freq=0.50,
    label_margin=0.05,
    title=None,
    title_fontsize=20,
    lw_exact=3,
    lw_pred=2,
    color_exact="black",
):

    color_preds = [
        "red", "blue", "green", "orange", "purple",
        "brown", "pink", "gray", "cyan", "magenta",
    ]

    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    length = u_true.shape[1]
    t_index = int((t - T_MIN) * length / (T_MAX - T_MIN))
    t_index = max(0, min(t_index, length - 1))

    ut = u_true[:, t_index]
    ax.plot(x, ut, linestyle="-", color=color_exact, linewidth=lw_exact)

    if not isinstance(u_preds, dict):
        u_preds = {"alg": u_preds}

    for alg, u_pred in u_preds.items():
        up = u_pred[:, t_index]
        ax.plot(x, up, linestyle="--", color=color_preds.pop(0), linewidth=lw_pred)

    ax.set_xlabel(xlabel, fontsize=ax_label_fontsize, labelpad=0)
    ax.set_ylabel(ylabel, fontsize=ax_label_fontsize, labelpad=0)

    xticks, xtick_labels = create_tick_labels(
        X_MIN, X_MAX, xtick_freq, decimal_points=2
    )
    yticks, ytick_labels = create_tick_labels(
        u_min, u_max, ytick_freq, decimal_points=2
    )

    ax.set_xticks(ticks=xticks, labels=xtick_labels, fontsize=ax_tick_fontsize)
    ax.set_yticks(ticks=yticks, labels=ytick_labels, fontsize=ax_tick_fontsize)
    ax.tick_params(axis="both", which="both", direction="out")
    ax.tick_params(axis="both", which="major", length=4, width=1.2)
    ax.tick_params(axis="both", which="minor", length=2, width=1)

    xmin = X_MIN * (1 + label_margin)
    xmax = X_MAX * (1 + label_margin)
    ymin = u_min * (1 + label_margin)
    ymax = u_max * (1 + label_margin)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if title:
        ax.set_title(title, fontsize=title_fontsize)

    if path_fig:
        plt.tight_layout()
        plt.savefig(path_fig, dpi=600, bbox_inches="tight")
        plt.close()

    return ax


def plot_model_results(
    X, y_true, y_pred, residuals, dir_plots, heatmap=True, slices=True
):

    Exact = y_true.reshape(len(np.unique(X[:, 1])), len(np.unique(X[:, 0]))).T
    y_pred = y_pred.reshape(len(np.unique(X[:, 1])), len(np.unique(X[:, 0]))).T

    # compute min and max across both exact and predicted
    u_min = min(Exact.min(), y_pred.min())
    u_max = max(Exact.max(), y_pred.max())
    print(f"u_min = {u_min:.6f}, u_max = {u_max:.6f}")

    m, n = Exact.shape
    t = np.linspace(0, 1, n).reshape(-1, 1)  # T x 1
    x = np.linspace(-1, 1, m).reshape(-1, 1)  # N x 1

    if heatmap:
        plot_heatmap(
            u_xt=Exact,
            cbar_label=r"$u(x, t)$",
            path_fig=os.path.join(dir_plots, "heatmap_exact.pdf"),
            figsize=(8, 4),
        )
        plot_heatmap(
            u_xt=y_pred,
            cbar_label=r"$\widehat{u}(x, t)$",
            path_fig=os.path.join(dir_plots, "heatmap_preds.pdf"),
            figsize=(8, 4),
        )
        plot_heatmap(
            u_xt=np.abs(Exact - y_pred),
            cbar_label=r"$|u(x, t) - \widehat{u}(x, t)|$",
            path_fig=os.path.join(dir_plots, "heatmap_abs_err.pdf"),
            c_min=0,
            c_max=2,
            figsize=(8, 4),
        )
        plot_heatmap(
            u_xt=residuals,
            cbar_label=r"$r$",
            path_fig=os.path.join(dir_plots, "heatmap_residual.pdf"),
            c_min=None,
            c_max=None,
            figsize=(8, 4),
        )

    if slices:
        for tval in (0, 0.25, 0.50, 0.75, 1.0):
            plot_slice(
                Exact,
                y_pred,
                x,
                t=tval,
                u_min=u_min,
                u_max=u_max,
                path_fig=os.path.join(dir_plots, f"slice_t{tval}.pdf"),
                title=f"t = {tval:.2f}",
            )
