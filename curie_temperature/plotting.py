import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linestyle": "--",
        "figure.dpi": 300,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
    }
)


def plot_rt(T, V, label, marker="o", linestyle="-"):
    plt.plot(
        T,
        V,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.4,
        markersize=4,
        label=label,
    )


def plot_derivative(T, dVdT, label, marker="s", linestyle="--"):
    plt.plot(
        T,
        dVdT,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.3,
        markersize=3,
        label=label,
    )


def plot_vtotal_fit(temperature, vtotal, vtotal_fit, label):
    plt.scatter(
        temperature,
        vtotal,
        marker="o",
        s=28,
        label=f"{label} data",
    )
    plt.plot(
        [min(temperature), max(temperature)],
        [vtotal_fit, vtotal_fit],
        linestyle="--",
        linewidth=1.4,
        label=f"{label} fit: y = {vtotal_fit:.4f} V",
    )

def finalize_plot(title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
