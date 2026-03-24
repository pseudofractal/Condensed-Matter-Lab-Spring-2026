import json
import os
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pint import UnitRegistry
from scipy.optimize import curve_fit
from uncertainties import ufloat
import shutil

# Initialize Unit Registry and satisfy type checkers
ureg = UnitRegistry()
Quantity: Any = ureg.Quantity

# Plotting Configuration
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


def get_paths() -> dict[str, Path]:
  """Initializes the directory structure relative to the script location."""
  base = Path(__file__).parent
  paths = {
    "data": base / "data",
    "plots": base / "plots",
    "final": base / "final",
    "sample": base / "sample_data",
    "config": base / "config.json",
  }
  for name, path in paths.items():
    if name != "config":
      path.mkdir(parents=True, exist_ok=True)
  return paths


def setup_experiment(paths: dict[str, Path]):
  """Sets up default configuration with per-material currents and sample data."""
  if not paths["config"].exists():
    config = {
      "system_resistance_ohm": 10.0,
      "calibration": {
        "material": "Silver",
        "current_A": 2.5,
        "mass_kg": 0.200,
        "cv_j_kgk": 235.0,
      },
      "samples": [
        {"material": "Copper", "current_A": 2.0, "mass_kg": 0.150},
        {"material": "Aluminum", "current_A": 3.0, "mass_kg": 0.120},
        {"material": "Iron", "current_A": 2.2, "mass_kg": 0.180},
      ],
    }
    with open(paths["config"], "w", encoding="utf-8") as f:
      json.dump(config, f, indent=2)

  cal_path = paths["sample"] / "calibration_silver.csv"
  if not cal_path.exists():
    pd.DataFrame({"dT_K": [2.0, 4.0, 6.0, 8.0], "dt_s": [45, 91, 136, 182]}).to_csv(
      cal_path, index=False
    )

  for name in ["copper", "aluminum", "iron"]:
    p = paths["sample"] / f"{name}_data.csv"
    if not p.exists():
      pd.DataFrame({"dT_K": [], "dt_s": []}).to_csv(p, index=False)


def linear_model(x: np.ndarray, m: float, c: float) -> np.ndarray:
  """Standard linear response model."""
  return m * x + c


LIT_VALUES = {
  "Silver": 235.0,
  "Copper": 385.0,
  "Aluminum": 900.0,
  "Brass": 380.0,
}


def perform_interval_analysis(
  df: pd.DataFrame, temp_min: float = 25.5, temp_max: float = 27.5
) -> dict[str, Any]:
  """Calculates statistics for intervals within a specific temperature range."""
  # Filter data for calculation
  mask = (df["Temp_C"] >= temp_min) & (df["Temp_C"] <= temp_max)
  calc_df = df[mask].copy()

  if calc_df.empty:
    # If range is completely missing, use first 5 points as fallback
    calc_df = df.head(5).copy()
    temp_min, temp_max = calc_df["Temp_C"].min(), calc_df["Temp_C"].max()

  intervals = calc_df["dt_s"].to_numpy()
  mean_dt = np.mean(intervals)
  std_dt = np.std(intervals, ddof=1)
  sem_dt = std_dt / np.sqrt(len(intervals)) if len(intervals) > 0 else np.nan

  return {
    "mean_dt": ufloat(mean_dt, std_dt),
    "std_dt": float(std_dt),
    "sem_dt": float(sem_dt),
    "all_x": df["Temp_C"].to_numpy(),
    "all_y": df["dt_s"].to_numpy(),
    "n": len(intervals),
    "range_str": f"[{temp_min}, {temp_max}]",
  }


def plot_simple_scatter(
  res: dict[str, Any],
  title: str,
  output_path: Path,
  result_text: str,
):
  """Simple scatter plot of all data with a result text box."""
  fig, ax = plt.subplots(figsize=(8, 5))

  ax.scatter(res["all_x"], res["all_y"], color="black", marker="o", s=30, label="Measured Data")

  ax.set_xlabel(r"Temperature $T$ [$^\circ$C]")
  ax.set_ylabel(r"Time to raise Temperature by 0.5 K [s]")
  ax.set_title(title)

  ax.text(
    0.05,
    0.95,
    result_text,
    transform=ax.transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    fontsize=11,
  )

  ax.grid(True, alpha=0.3, linestyle="--")
  plt.tight_layout()
  plt.savefig(output_path)
  plt.close()


def plot_cumulative_silver(df: pd.DataFrame, output_path: Path):
  """Plots cumulative time vs temperature for Silver with multiple curve fits."""
  times = np.cumsum(df["dt_s"].to_numpy())
  temps = df["Temp_C"].to_numpy()

  fig, ax = plt.subplots(figsize=(8, 6))
  ax.scatter(temps, times, color="black", s=30, label="Data Points", zorder=10)

  x_smooth = np.linspace(temps.min(), temps.max(), 200)

  # Fits: 2, 3, 4, 5, 10
  degrees = [2, 3, 5, 10]
  colors = ["red", "green", "blue", "orange"]
  styles = ["--", "-", "-.", ":"]

  for deg, color, style in zip(degrees, colors, styles):
    p = np.poly1d(np.polyfit(temps, times, deg))
    ax.plot(
      x_smooth, p(x_smooth), color=color, linestyle=style, alpha=0.8, label=f"Degree {deg} Fit"
    )

  ax.set_xlabel(r"Temperature $T$ [$^\circ$C]")
  ax.set_ylabel(r"Total Time Elapsed $t$ [s]")
  ax.set_title("Silver Calibration: Cumulative Time vs Temperature (High Order Fits)")
  ax.grid(True, alpha=0.3, linestyle="--")
  ax.legend(loc="upper left")

  plt.tight_layout()
  plt.savefig(output_path)
  plt.close()


def plot_silver_residuals(df: pd.DataFrame, output_path: Path):
  """Plots residuals for various polynomial degrees on Silver data."""
  times = np.cumsum(df["dt_s"].to_numpy())
  temps = df["Temp_C"].to_numpy()

  fig, ax = plt.subplots(figsize=(8, 6))

  degrees = [2, 3, 5, 10]
  colors = ["red", "green", "blue", "orange"]
  markers = ["o", "s", "D", "x"]

  for deg, color, marker in zip(degrees, colors, markers):
    p = np.poly1d(np.polyfit(temps, times, deg))
    res = times - p(temps)
    ax.plot(
      temps,
      res,
      color=color,
      marker=marker,
      linestyle="-",
      alpha=0.7,
      label=f"Degree {deg} Residuals",
    )

  ax.axhline(0, color="black", linewidth=1, linestyle="-")
  ax.set_xlabel(r"Temperature $T$ [$^\circ$C]")
  ax.set_ylabel(r"Residual $(t_{obs} - t_{pred})$ [s]")
  ax.set_title("Residual Analysis: High-Order Polynomial Fits")
  ax.grid(True, alpha=0.3, linestyle="--")
  ax.legend()

  plt.tight_layout()
  plt.savefig(output_path)
  plt.close()


def generate_latex_output(
  paths: dict[str, Path], summary: list[dict], all_data: dict[str, pd.DataFrame]
):
  """Generates LaTeX tables for results and raw data, converted to images."""
  # Resolve tool paths if not on PATH
  pdflatex_cmd = shutil.which("pdflatex")
  if pdflatex_cmd is None:
    candidate = Path(r"C:\Users\Dell\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe")
    if candidate.exists():
      pdflatex_cmd = str(candidate)

  magick_cmd = shutil.which("magick")
  if magick_cmd is None:
    candidate = Path(r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe")
    if candidate.exists():
      magick_cmd = str(candidate)

  latex_cmd = shutil.which("latex")
  if latex_cmd is None:
    candidate = Path(r"C:\Users\Dell\AppData\Local\Programs\MiKTeX\miktex\bin\x64\latex.exe")
    if candidate.exists():
      latex_cmd = str(candidate)

  dvipng_cmd = shutil.which("dvipng")
  if dvipng_cmd is None:
    candidate = Path(r"C:\Users\Dell\AppData\Local\Programs\MiKTeX\miktex\bin\x64\dvipng.exe")
    if candidate.exists():
      dvipng_cmd = str(candidate)

  if pdflatex_cmd is None and latex_cmd is None:
    print("LaTeX tools not found. Skipping PNG table generation.")
    return

  # 1. Results Table
  res_tex = paths["final"] / "results_table.tex"
  res_content = [
    r"\documentclass[varwidth,border=10pt]{standalone}",
    r"\usepackage{booktabs, array, amsmath}",
    r"\begin{document}",
    r"\centering \small \textbf{Calorimetry Results Summary}\\[1ex]",
    r"\begin{tabular}{lccc}",
    r"\toprule",
    r"Material & Mass [g] & Current [A] & $C_v$ [J/kg$\cdot$K] \\",
    r"\midrule",
  ]

  for item in summary:
    res_content.append(
      f"{item['Material']} & {item['Mass'] * 1e3:.1f} & {item['Current']:.3f} & "
      f"{item['Cv'].n:.1f} $\pm$ {item['Cv'].s:.1f} \\\\"
    )

  res_content.extend([r"\bottomrule", r"\end{tabular}", r"\end{document}"])

  with open(res_tex, "w") as f:
    f.write("\n".join(res_content))

  # 2. Raw Data Table (Side-by-side)
  data_tex = paths["final"] / "data_table.tex"

  # Align materials: Ag, Cu, Al, Brass
  order = ["Silver", "Copper", "Aluminum", "Brass"]
  cols = []
  for mat in order:
    if mat in all_data:
      cols.append(all_data[mat])

  max_rows = max(len(df) for df in cols) if cols else 0

  data_content = [
    r"\documentclass[varwidth,border=15pt]{standalone}",
    r"\usepackage{booktabs, array}",
    r"\begin{document}",
    r"\centering \footnotesize \textbf{Experimental Heating Intervals}\\[1ex]",
    r"\begin{tabular}{" + "cc|" * (len(cols) - 1) + "cc}",
    r"\toprule",
  ]

  header1 = (
    " & ".join([rf"\multicolumn{{2}}{{c}}{{{mat}}}" for mat in order if mat in all_data])
    + r" \\"
  )
  header2 = (
    " & ".join([r"$T$ [$^\circ$C] & $\Delta t$ [s]" for mat in order if mat in all_data])
    + r" \\"
  )

  data_content.extend([header1, header2, r"\midrule"])

  for i in range(max_rows):
    row_parts = []
    for df in cols:
      if i < len(df):
        row_parts.append(f"{df.iloc[i]['Temp_C']:.1f} & {df.iloc[i]['dt_s']:.2f}")
      else:
        row_parts.append("&")
    data_content.append(" & ".join(row_parts) + r" \\")

  data_content.extend([r"\bottomrule", r"\end{tabular}", r"\end{document}"])

  with open(data_tex, "w") as f:
    f.write("\n".join(data_content))

  # Compilation
  for tex_file, out_name in [(res_tex, "results_summary.png"), (data_tex, "data_table.png")]:
    try:
      if pdflatex_cmd is None or magick_cmd is None:
        raise RuntimeError("pdflatex/magick not available")

      subprocess.run(
        [
          pdflatex_cmd,
          "-interaction=nonstopmode",
          "-output-directory",
          str(paths["final"]),
          str(tex_file),
        ],
        check=True,
        capture_output=True,
      )
      subprocess.run(
        [
          magick_cmd,
          "-density",
          "300",
          str(tex_file.with_suffix(".pdf")),
          str(paths["final"] / out_name),
        ],
        check=True,
      )
    except Exception as e:
      # Fallback: latex -> dvi -> dvipng (no Ghostscript needed)
      if latex_cmd and dvipng_cmd:
        try:
          subprocess.run(
            [
              latex_cmd,
              "-interaction=nonstopmode",
              "-output-directory",
              str(paths["final"]),
              str(tex_file),
            ],
            check=True,
            capture_output=True,
          )
          subprocess.run(
            [
              dvipng_cmd,
              "-D",
              "300",
              "-T",
              "tight",
              "-o",
              str(paths["final"] / out_name),
              str(tex_file.with_suffix(".dvi")),
            ],
            check=True,
            capture_output=True,
          )
        except Exception as e2:
          print(f"Latex/DVIPNG error for {tex_file.name}: {e2}")
      else:
        print(f"Latex/Magick error for {tex_file.name}: {e}")
    finally:
      # Cleanup
      for ext in [".aux", ".log", ".pdf", ".tex", ".dvi"]:
        p = tex_file.with_suffix(ext)
        if p.exists():
          os.remove(p)


def main():
  paths = get_paths()
  setup_experiment(paths)

  with open(paths["config"], encoding="utf-8") as f:
    cfg = json.load(f)

  resistance = cfg["system_resistance_ohm"]

  # Phase 1: Calibration
  cal_cfg = cfg["calibration"]
  cal_file = paths["data"] / f"calibration_{cal_cfg['material'].lower()}.csv"

  if not cal_file.exists():
    print(f"Phase 1 missing. Provide calibration data at {cal_file}")
    return

  df_cal = pd.read_csv(cal_file)
  res_cal = perform_interval_analysis(df_cal, temp_min=25.5, temp_max=27.5)
  all_data = {"Silver": df_cal}

  # Physics: m_slope = mean_dt / 0.5
  effective_slope_cal = res_cal["mean_dt"] / 0.5
  cal_power = (cal_cfg["current_A"] ** 2) * resistance
  water_equivalent = effective_slope_cal * cal_power - (cal_cfg["mass_kg"] * cal_cfg["cv_j_kgk"])

  plot_simple_scatter(
    res_cal,
    f"Calibration Analysis (Silver)",
    paths["plots"] / "calibration-Ag.png",
    rf"Water Equivalent (W):" "\n" rf"${water_equivalent.n:.2f} \pm {water_equivalent.s:.2f}$ J/K",
  )

  # New Cumulative Plot for Silver
  plot_cumulative_silver(df_cal, paths["plots"] / "calibration-Ag-cumulative.png")
  plot_silver_residuals(df_cal, paths["plots"] / "calibration-Ag-residuals.png")

  print(f"Calibration Complete. W = {water_equivalent:.2f} J/K")

  # Phase 2: Iterating through Materials
  summary_results = []
  found_phase2 = False

  for sample in cfg["samples"]:
    mat_name = sample["material"]
    mat_file = paths["data"] / f"{mat_name.lower()}_data.csv"

    if not mat_file.exists() or pd.read_csv(mat_file).empty:
      continue

    found_phase2 = True
    df_mat = pd.read_csv(mat_file)
    all_data[mat_name] = df_mat
    res_mat = perform_interval_analysis(df_mat, temp_min=25.5, temp_max=27.5)

    # Physics: Cv = (m_slope * I^2 * R - W) / m_sample
    effective_slope_mat = res_mat["mean_dt"] / 0.5
    mat_power = (sample["current_A"] ** 2) * resistance
    cv_val = (effective_slope_mat * mat_power - water_equivalent) / sample["mass_kg"]

    # Mapping material names for files
    file_map = {"Aluminum": "Al", "Copper": "Cu", "Brass": "Brass"}
    file_name = f"{file_map.get(mat_name, mat_name)}.png"

    plot_simple_scatter(
      res_mat,
      f"Specific Heat Analysis: {mat_name}",
      paths["plots"] / file_name,
      rf"$C_v = {cv_val.n:.2f} \pm {cv_val.s:.2f}$ J/(kg·K)",
    )

    summary_results.append(
      {
        "Material": mat_name,
        "Current": sample["current_A"],
        "Mass": sample["mass_kg"],
        "Cv": cv_val,
        "n": res_mat["n"],
        "std_dt": res_mat["std_dt"],
        "sem_dt": res_mat["sem_dt"],
      }  # Stats for mean-based method
    )

  if not found_phase2:
    print("Phase 1 finished. Populate sample data files for Phase 2.")
    return

  # Generate Final Report
  report_path = paths["final"] / "experiment_summary.txt"
  generate_latex_output(paths, summary_results, all_data)

  with open(report_path, "w", encoding="utf-8") as f:
    f.write("Calorimetry Analysis Report\n" + "=" * 40 + "\n")
    f.write(f"Water Equivalent (W): {water_equivalent.n:.4e} +/- {water_equivalent.s:.4e} J/K\n")
    f.write("-" * 40 + "\n")
    for item in summary_results:
      lit = LIT_VALUES.get(item["Material"], 0)
      exp_val = item["Cv"].n
      error = ((exp_val - lit) / lit) * 100 if lit != 0 else 0
      f.write(f"Material: {item['Material']}\n")
      f.write(f"  Applied Current: {item['Current']:.2f} A\n")
      f.write(f"  Data Points Used (n): {item['n']}\n")
      f.write(f"  Std dev of dt: {item['std_dt']:.4e} s\n")
      f.write(f"  Std error of mean dt: {item['sem_dt']:.4e} s\n")
      f.write(f"  Measured Cv:     {item['Cv'].n:.4e} +/- {item['Cv'].s:.4e} J/(kg·K)\n")
      f.write(f"  Literature Cv:   {lit} J/(kg·K)\n")
      f.write(f"  Relative Error:  {error:.2f}%\n")
      f.write("\n")

  print(f"Analysis complete for {len(summary_results)} materials. Results in {report_path}")


if __name__ == "__main__":
  main()
