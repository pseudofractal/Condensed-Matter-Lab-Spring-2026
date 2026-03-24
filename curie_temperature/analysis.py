import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from config import (
    SMOOTH_WINDOW,
    SMOOTH_POLYORDER,
    TEMP_LEAST_COUNT_C,
    VOLTAGE_LEAST_COUNT_V,
    MC_SAMPLES,
)
from utils import (
    load_data,
    apply_vtotal_method,
    compute_capacitance,
    compute_transition_temperature,
)
from plotting import plot_rt, plot_derivative, finalize_plot, plot_vtotal_fit

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
TABLES_DIR = os.path.join(BASE_DIR, "tables")
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "0") == "1"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

FILES = {
    "SC1 (22 nF)": "SC1_22nF_raw_data.csv",
    "SC2 (39 nF)": "SC2_39nF_raw_data.csv",
    "SC3 (62 nF)": "SC3_62nF_raw_data.csv",
}

results = {}
vtotal_fit_points = {}
vtotal_fit_values = {}
transition_results = {}
summary_rows = []
vtotal_rows = []


def slugify(label):
    return re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_").lower()


def clear_output_dir(folder):
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            os.remove(path)


def cleanup_legacy_root_outputs():
    legacy_files = [
        "epsilon_r_vs_T_all.png",
        "d_epsilon_r_dT_vs_T_all.png",
        "vtotal_constant_fit.png",
        "transition_temperature_report.txt",
    ]
    for filename in legacy_files:
        legacy_path = os.path.join(BASE_DIR, filename)
        if os.path.exists(legacy_path):
            os.remove(legacy_path)


def _render_table_image(df, title, out_path):
    n_rows = len(df)
    fig_h = max(3.0, 0.45 * (n_rows + 2))
    fig_w = min(20.0, max(10.0, 1.6 * len(df.columns)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=8)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def prettify_columns(df):
    pretty_map = {
        "Temperature_C": "Temperature (°C)",
        "Vdc_V": "Vdc (V)",
        "Vsc_V": "Vsc (V)",
        "C": "C = Vsc / Vdc",
        "epsilon_r": "εr",
        "epsilon_r_smooth": "εr (smoothed)",
        "d_epsilon_r_dT": "dεr/dT (1/°C)",
        "sample": "Sample",
        "Tc_peak_degC": "Tc from εr peak (°C)",
        "Tc_derivative_degC": "Tc from max dεr/dT (°C)",
        "Tc_combined_degC": "Combined Tc (°C)",
        "Tc_peak_unc_degC": "Uncertainty in Tc from εr peak (°C)",
        "Tc_derivative_unc_degC": "Uncertainty in Tc from max dεr/dT (°C)",
        "Tc_combined_unc_degC": "Combined Tc uncertainty (°C)",
        "Tc_combined_unc_pct": "Combined uncertainty (%)",
        "Tc_method_difference_degC": "|Tc(εr peak) - Tc(max dεr/dT)| (°C)",
        "epsilon_noise_std": "Std dev of εr smoothing residual",
        "vtotal_fit_V": "Vtotal fit (V)",
        "C_ref": "Reference C",
        "n_points_total": "Total data points",
        "n_points_vtotal_fit": "Points used in Vtotal fit",
    }
    out = df.copy()
    out.columns = [pretty_map.get(c, c.replace("_", " ")) for c in out.columns]
    return out


def save_stacked_table_image(df, title, filename, folder):
    _render_table_image(prettify_columns(df), title, os.path.join(folder, filename))


clear_output_dir(RESULTS_DIR)
clear_output_dir(PLOTS_DIR)
clear_output_dir(TABLES_DIR)
cleanup_legacy_root_outputs()

for label, filename in FILES.items():

    path = os.path.join(DATA_DIR, filename)

    df = load_data(path)

    df, V_total_fit, valid = apply_vtotal_method(df)
    df, C_ref = compute_capacitance(df)
    transition = compute_transition_temperature(
        df,
        smooth_window=SMOOTH_WINDOW,
        smooth_polyorder=SMOOTH_POLYORDER,
        temp_least_count_c=TEMP_LEAST_COUNT_C,
        voltage_least_count_v=VOLTAGE_LEAST_COUNT_V,
        mc_samples=MC_SAMPLES,
    )

    results[label] = df
    vtotal_fit_points[label] = valid
    vtotal_fit_values[label] = V_total_fit
    transition_results[label] = transition
    vtotal_rows.append(
        {
            "sample": label,
            "vtotal_fit_V": float(V_total_fit),
            "C_ref": float(C_ref),
            "n_points_total": int(len(df)),
            "n_points_vtotal_fit": int(len(valid)),
        }
    )

    sample_slug = slugify(label)
    processed_df = df.sort_values("Temperature_C").reset_index(drop=True).copy()
    processed_df["epsilon_r_smooth"] = transition["epsilon_smooth"]
    processed_df["d_epsilon_r_dT"] = transition["d_eps_dT"]
    numeric_cols = processed_df.select_dtypes(include="number").columns
    processed_df[numeric_cols] = processed_df[numeric_cols].round(6)
    save_stacked_table_image(
        processed_df,
        title=f"{label} Processed Data Table",
        filename=f"{sample_slug}_processed_data_table.png",
        folder=TABLES_DIR,
    )

    summary_rows.append(
        {
            "sample": label,
            "Tc_peak_degC": transition["Tc_peak"],
            "Tc_peak_unc_degC": transition["Tc_peak_unc"],
            "Tc_derivative_degC": transition["Tc_derivative"],
            "Tc_derivative_unc_degC": transition["Tc_derivative_unc"],
            "Tc_combined_degC": transition["Tc_combined"],
            "Tc_combined_unc_degC": transition["Tc_combined_unc"],
            "Tc_combined_unc_pct": transition["Tc_combined_unc_pct"],
            "Tc_method_difference_degC": transition["Tc_method_difference"],
            "epsilon_noise_std": transition["epsilon_noise_std"],
            "epsilon_instr_unc_mean": transition["epsilon_instr_unc_mean"],
            "Tc_peak_mc_std": transition["Tc_peak_mc_std"],
            "Tc_derivative_mc_std": transition["Tc_derivative_mc_std"],
        }
    )

    print(f"\n{label}")
    print("V_total_fit =", V_total_fit)
    print("Reference C =", C_ref)
    print(
        f"Transition temperature Tc (peak epsilon_r) = "
        f"{transition['Tc_peak']:.2f} +/- {transition['Tc_peak_unc']:.2f} degC"
    )
    print(
        f"Transition temperature Tc (max d epsilon_r/dT) = "
        f"{transition['Tc_derivative']:.2f} +/- {transition['Tc_derivative_unc']:.2f} degC"
    )
    print(
        f"Combined Tc (with method spread) = "
        f"{transition['Tc_combined']:.2f} +/- {transition['Tc_combined_unc']:.2f} degC"
    )
    print(
        f"Instrument settings: dT = {TEMP_LEAST_COUNT_C:.2f} degC, "
        f"dV = {VOLTAGE_LEAST_COUNT_V:.2f} V, MC samples = {MC_SAMPLES}"
    )

# -----------------------
# Plot epsilon_r vs T
# -----------------------

plt.figure()
styles = [("o", "-"), ("s", "--"), ("^", "-.")]

for (label, df), (marker, linestyle) in zip(results.items(), styles):
    plot_rt(df["Temperature_C"], df["epsilon_r"], label=label, marker=marker, linestyle=linestyle)
    transition = transition_results[label]
    plt.scatter(
        transition["Tc_peak"],
        transition["epsilon_at_peak"],
        marker="X",
        s=60,
        label=(
            f"{label} Tc (εr peak) = {transition['Tc_peak']:.2f} "
            f"+/- {transition['Tc_peak_unc']:.2f} °C"
        ),
    )

finalize_plot(
    title=r"Relative Permittivity $\epsilon_r$ vs Temperature",
    xlabel=r"Temperature ($^\circ$C)",
    ylabel=r"Relative Permittivity $\epsilon_r$",
)
plt.savefig(os.path.join(PLOTS_DIR, "epsilon_r_vs_T_all.png"))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

# -----------------------
# Plot epsilon_r vs T (offset for visual separation)
# -----------------------

plt.figure()
offset_step = 0.35

for idx, ((label, df), (marker, linestyle)) in enumerate(zip(results.items(), styles)):
    offset = idx * offset_step
    y_vals = df["epsilon_r"] + offset
    plot_rt(
        df["Temperature_C"],
        y_vals,
        label=f"{label} (+{offset:.2f})",
        marker=marker,
        linestyle=linestyle,
    )
    transition = transition_results[label]
    plt.scatter(
        transition["Tc_peak"],
        transition["epsilon_at_peak"] + offset,
        marker="X",
        s=60,
        label=(
            f"{label} Tc (offset) = {transition['Tc_peak']:.2f} "
            f"+/- {transition['Tc_peak_unc']:.2f} Â°C"
        ),
    )

finalize_plot(
    title=r"Relative Permittivity $\epsilon_r$ vs Temperature (Offset Curves)",
    xlabel=r"Temperature ($^\circ$C)",
    ylabel=r"Relative Permittivity $\epsilon_r$ + offset",
)
plt.savefig(os.path.join(PLOTS_DIR, "epsilon_r_vs_T_all_offset.png"))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

# -----------------------
# Plot d(epsilon_r)/dT vs T with Tc markers
# -----------------------

plt.figure()
for (label, transition), (marker, linestyle) in zip(transition_results.items(), styles):
    plot_derivative(
        transition["temperature"],
        transition["d_eps_dT"],
        label=label,
        marker=marker,
        linestyle=linestyle,
    )
    plt.axvline(
        transition["Tc_derivative"],
        linestyle=":",
        linewidth=1.0,
        alpha=0.7,
        label=(
            f"{label} Tc (max dεr/dT) = {transition['Tc_derivative']:.2f} "
            f"+/- {transition['Tc_derivative_unc']:.2f} °C"
        ),
    )
    plt.scatter(
        transition["Tc_derivative"],
        transition["d_eps_dT_at_peak"],
        marker="D",
        s=40,
    )

finalize_plot(
    title=r"$d\epsilon_r/dT$ vs Temperature",
    xlabel=r"Temperature ($^\circ$C)",
    ylabel=r"$d\epsilon_r/dT$ (1/$^\circ$C)",
)
plt.savefig(os.path.join(PLOTS_DIR, "d_epsilon_r_dT_vs_T_all.png"))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

# -----------------------
# Plot V_total constant fit
# -----------------------

plt.figure()
for label, valid in vtotal_fit_points.items():
    vtotal_values = valid["Vdc_V"] + valid["Vsc_V"]
    vtotal_fit = vtotal_fit_values[label]
    plot_vtotal_fit(valid["Temperature_C"], vtotal_values, vtotal_fit, label)

finalize_plot(
    title=r"Constant Fit for $V_{total}$ using valid points",
    xlabel=r"Temperature ($^\circ$C)",
    ylabel=r"$V_{total}$ (V)",
)
plt.savefig(os.path.join(PLOTS_DIR, "vtotal_constant_fit.png"))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

summary_df = pd.DataFrame(summary_rows)
summary_image_df = summary_df[
    [
        "sample",
        "Tc_peak_degC",
        "Tc_derivative_degC",
        "Tc_combined_degC",
    ]
].copy()
summary_image_df = summary_image_df.round(3)
save_stacked_table_image(
    summary_image_df,
    title="Curie Temperature Summary",
    filename="curie_temperature_summary_table.png",
    folder=TABLES_DIR,
)

error_df = summary_df[
    [
        "sample",
        "Tc_peak_unc_degC",
        "Tc_derivative_unc_degC",
        "Tc_combined_unc_degC",
        "Tc_combined_unc_pct",
        "Tc_method_difference_degC",
        "epsilon_noise_std",
        "epsilon_instr_unc_mean",
        "Tc_peak_mc_std",
        "Tc_derivative_mc_std",
    ]
].copy()
error_df = error_df.round(4)
save_stacked_table_image(
    error_df,
    title="Curie Temperature Error Analysis",
    filename="curie_temperature_error_analysis_table.png",
    folder=TABLES_DIR,
)

vtotal_df = pd.DataFrame(vtotal_rows).round(6)
save_stacked_table_image(
    vtotal_df,
    title="Vtotal Fit Summary",
    filename="vtotal_fit_summary_table.png",
    folder=TABLES_DIR,
)

with open(os.path.join(RESULTS_DIR, "transition_temperature_report.txt"), "w", encoding="utf-8") as f:
    f.write("Transition Temperature (Curie) Report\n")
    f.write("=" * 42 + "\n")
    f.write(
        f"Uncertainty inputs: dT = {TEMP_LEAST_COUNT_C:.3f} degC, "
        f"dVdc = dVsc = {VOLTAGE_LEAST_COUNT_V:.3f} V, MC samples = {MC_SAMPLES}\n"
    )
    f.write("=" * 42 + "\n")
    for label, transition in transition_results.items():
        f.write(f"{label}\n")
        f.write(
            f"Tc from peak epsilon_r: {transition['Tc_peak']:.2f} +/- "
            f"{transition['Tc_peak_unc']:.2f} degC\n"
        )
        f.write(
            f"Tc from max d(epsilon_r)/dT: {transition['Tc_derivative']:.2f} +/- "
            f"{transition['Tc_derivative_unc']:.2f} degC\n"
        )
        f.write(
            f"Combined Tc: {transition['Tc_combined']:.2f} +/- "
            f"{transition['Tc_combined_unc']:.2f} degC "
            f"({transition['Tc_combined_unc_pct']:.2f}% relative)\n"
        )
        f.write(
            f"Method difference |Tc_peak - Tc_derivative|: "
            f"{transition['Tc_method_difference']:.2f} degC\n"
        )
        f.write(
            f"Epsilon smoothing residual std: {transition['epsilon_noise_std']:.4e}\n"
        )
        f.write(
            f"Mean propagated epsilon instrument uncertainty: "
            f"{transition['epsilon_instr_unc_mean']:.4e}\n"
        )
        f.write(
            f"MC Tc std (peak epsilon_r): {transition['Tc_peak_mc_std']:.4f} degC\n"
        )
        f.write(
            f"MC Tc std (max d(epsilon_r)/dT): {transition['Tc_derivative_mc_std']:.4f} degC\n"
        )
        f.write("-" * 42 + "\n")

json_payload = {
    label: {
        "Tc_peak_degC": transition["Tc_peak"],
        "Tc_peak_unc_degC": transition["Tc_peak_unc"],
        "Tc_derivative_degC": transition["Tc_derivative"],
        "Tc_derivative_unc_degC": transition["Tc_derivative_unc"],
        "Tc_combined_degC": transition["Tc_combined"],
        "Tc_combined_unc_degC": transition["Tc_combined_unc"],
        "Tc_combined_unc_pct": transition["Tc_combined_unc_pct"],
        "Tc_method_difference_degC": transition["Tc_method_difference"],
        "epsilon_noise_std": transition["epsilon_noise_std"],
        "epsilon_instr_unc_mean": transition["epsilon_instr_unc_mean"],
        "Tc_peak_mc_std": transition["Tc_peak_mc_std"],
        "Tc_derivative_mc_std": transition["Tc_derivative_mc_std"],
    }
    for label, transition in transition_results.items()
}

with open(os.path.join(RESULTS_DIR, "transition_temperature_report.json"), "w", encoding="utf-8") as f:
    json.dump(json_payload, f, indent=2)
