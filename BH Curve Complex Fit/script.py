import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

plt.rcParams.update(
  {
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "--",
    "figure.dpi": 300,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
  }
)


@dataclass
class TraceData:
  time: np.ndarray
  value: np.ndarray
  source: str
  vertical_scale: float
  vertical_offset: float
  file_name: str


def get_paths() -> dict[str, Path]:
  base = Path(__file__).parent
  paths = {
    "base": base,
    "raw": base / "raw",
    "plots": base / "plots",
    "final": base / "final",
    "config": base / "config.json",
  }
  for key in ("plots", "final"):
    paths[key].mkdir(parents=True, exist_ok=True)
  return paths


def load_config(config_path: Path) -> dict[str, Any]:
  with open(config_path, encoding="utf-8") as f:
    return json.load(f)


def read_tek_csv(path: Path) -> TraceData:
  df = pd.read_csv(path, header=None)
  source = str(df.iloc[6, 1]).strip()
  vertical_scale = float(df.iloc[8, 1])
  vertical_offset = float(df.iloc[9, 1])
  data = df.iloc[18:, 3:5].copy()
  data.columns = ["time", "value"]
  data = data.apply(pd.to_numeric, errors="coerce").dropna()
  return TraceData(
    time=data["time"].to_numpy(),
    value=data["value"].to_numpy(),
    source=source,
    vertical_scale=vertical_scale,
    vertical_offset=vertical_offset,
    file_name=path.name,
  )


def pair_traces(raw_dir: Path) -> list[dict[str, TraceData]]:
  files = sorted(raw_dir.glob("TEK*.CSV"))
  if len(files) % 2 != 0:
    raise ValueError("Expected an even number of TEK files for CH1/CH2 pairing.")

  pairs = []
  for idx in range(0, len(files), 2):
    first = read_tek_csv(files[idx])
    second = read_tek_csv(files[idx + 1])
    pair = {first.source: first, second.source: second}
    if set(pair) != {"CH1", "CH2"}:
      raise ValueError(f"Could not pair {files[idx].name} and {files[idx + 1].name}.")
    pairs.append(pair)
  return pairs


def estimate_frequency(time: np.ndarray, value: np.ndarray) -> float:
  centered = value - np.mean(value)
  dt = float(np.median(np.diff(time)))
  spectrum = np.fft.rfft(centered)
  freqs = np.fft.rfftfreq(len(centered), dt)
  mask = (freqs > 1.0) & (freqs < 500.0)
  if not np.any(mask):
    return 50.0
  return float(freqs[mask][np.argmax(np.abs(spectrum[mask]))])


def trig_design_matrix(time: np.ndarray, omega: float, order: int) -> np.ndarray:
  cols = [np.ones_like(time)]
  for n in range(1, order + 1):
    cols.append(np.sin(n * omega * time))
    cols.append(np.cos(n * omega * time))
  return np.column_stack(cols)


def fit_trig_series(time: np.ndarray, value: np.ndarray, order: int) -> dict[str, Any]:
  omega_guess = 2.0 * np.pi * estimate_frequency(time, value)

  def objective(omega: float) -> float:
    design = trig_design_matrix(time, omega, order)
    coeffs, *_ = np.linalg.lstsq(design, value, rcond=None)
    residual = value - design @ coeffs
    return float(np.mean(residual**2))

  result = minimize_scalar(
    objective,
    bounds=(0.7 * omega_guess, 1.3 * omega_guess),
    method="bounded",
  )
  omega = float(result.x)
  design = trig_design_matrix(time, omega, order)
  coeffs, *_ = np.linalg.lstsq(design, value, rcond=None)
  fitted = design @ coeffs
  residual = value - fitted
  ss_res = float(np.sum(residual**2))
  ss_tot = float(np.sum((value - np.mean(value)) ** 2))
  r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
  return {
    "omega": omega,
    "coeffs": coeffs,
    "frequency_hz": omega / (2.0 * np.pi),
    "fitted": fitted,
    "r2": r2,
  }


def evaluate_trig_series(time: np.ndarray, coeffs: np.ndarray, omega: float) -> np.ndarray:
  order = (len(coeffs) - 1) // 2
  return trig_design_matrix(time, omega, order) @ coeffs


def evaluate_trig_series_derivative(time: np.ndarray, coeffs: np.ndarray, omega: float) -> np.ndarray:
  order = (len(coeffs) - 1) // 2
  derivative = np.zeros_like(time)
  for n in range(1, order + 1):
    s = coeffs[2 * n - 1]
    c = coeffs[2 * n]
    derivative += n * omega * s * np.cos(n * omega * time)
    derivative -= n * omega * c * np.sin(n * omega * time)
  return derivative


def trig_equation_string(coeffs: np.ndarray, omega: float, var_name: str, keep_terms: int = 4) -> str:
  terms = []
  if abs(coeffs[0]) > 1e-12:
    terms.append((abs(coeffs[0]), f"{coeffs[0]:+.4g}"))
  order = (len(coeffs) - 1) // 2
  for n in range(1, order + 1):
    s = coeffs[2 * n - 1]
    c = coeffs[2 * n]
    terms.append((abs(s), f"{s:+.4g}\\sin({n}\\omega t)"))
    terms.append((abs(c), f"{c:+.4g}\\cos({n}\\omega t)"))

  chosen = [text for _, text in sorted(terms, key=lambda item: item[0], reverse=True)[: keep_terms + 1]]
  ordered = []
  const_term = f"{coeffs[0]:+.4g}"
  if const_term in chosen:
    ordered.append(const_term)
  for n in range(1, order + 1):
    s_term = f"{coeffs[2 * n - 1]:+.4g}\\sin({n}\\omega t)"
    c_term = f"{coeffs[2 * n]:+.4g}\\cos({n}\\omega t)"
    if s_term in chosen:
      ordered.append(s_term)
    if c_term in chosen:
      ordered.append(c_term)

  body = " ".join(ordered) if ordered else "0"
  return (
    rf"${var_name}(t) = {body}$"
    "\n"
    + rf"$\omega = {omega:.4g}\,\mathrm{{rad\,s^{{-1}}}},\quad f = {omega / (2.0 * np.pi):.4g}\,\mathrm{{Hz}}$"
  )


def polygon_area(x: np.ndarray, y: np.ndarray) -> float:
  return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def signed_polygon_area(x: np.ndarray, y: np.ndarray) -> float:
  return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def crossings(x: np.ndarray, y: np.ndarray, target: float = 0.0) -> list[float]:
  shifted = y - target
  values = []
  for i in range(len(shifted) - 1):
    y1 = shifted[i]
    y2 = shifted[i + 1]
    if y1 == 0:
      values.append(float(x[i]))
      continue
    if y1 * y2 < 0:
      frac = abs(y1) / (abs(y1) + abs(y2))
      values.append(float(x[i] + frac * (x[i + 1] - x[i])))
  return values


def split_by_sign(values: list[float]) -> tuple[list[float], list[float]]:
  pos = [value for value in values if value >= 0]
  neg = [value for value in values if value <= 0]
  return pos, neg


def compute_loop_metrics(h: np.ndarray, b: np.ndarray) -> dict[str, float | None]:
  area_bh = polygon_area(h, b)
  signed_area_bh = signed_polygon_area(h, b)

  b_at_h0 = crossings(b, h, 0.0)
  h_at_b0 = crossings(h, b, 0.0)
  b_pos, b_neg = split_by_sign(b_at_h0)
  h_pos, h_neg = split_by_sign(h_at_b0)

  remanence = None
  if b_pos and b_neg:
    remanence = 0.5 * (np.mean(np.abs(b_pos)) + np.mean(np.abs(b_neg)))

  coercivity = None
  if h_pos and h_neg:
    coercivity = 0.5 * (np.mean(np.abs(h_pos)) + np.mean(np.abs(h_neg)))

  mask = np.abs(h) <= 0.15 * np.max(np.abs(h))
  mu_initial = None
  if np.count_nonzero(mask) >= 5:
    slope, _ = np.polyfit(h[mask], b[mask], 1)
    mu_initial = float(abs(slope))

  return {
    "loop_area_bh": float(area_bh),
    "signed_loop_area_bh": float(signed_area_bh),
    "energy_loss_per_cycle_per_volume": float(area_bh),
    "remanence_B": None if remanence is None else float(remanence),
    "coercivity_H": None if coercivity is None else float(coercivity),
    "initial_slope_dBdH": mu_initial,
    "H_max": float(np.max(h)),
    "H_min": float(np.min(h)),
    "B_max": float(np.max(b)),
    "B_min": float(np.min(b)),
  }


def compute_parametric_loop_area(h: np.ndarray, db_dt: np.ndarray, time: np.ndarray) -> tuple[float, float]:
  signed_area = float(np.trapezoid(h * db_dt, time))
  return abs(signed_area), signed_area


def build_time_grid(time: np.ndarray, samples: int = 4000) -> np.ndarray:
  return np.linspace(time.min(), time.max(), samples, endpoint=False)


def analyze_pair(
  pair: dict[str, TraceData],
  meta: dict[str, Any],
  config: dict[str, Any],
  blank_slope: float | None,
) -> dict[str, Any]:
  ch1 = pair["CH1"]
  ch2 = pair["CH2"]
  order = int(config["fit"]["harmonic_order"])

  fit_vx = fit_trig_series(ch1.time, ch1.value, order)
  fit_vy = fit_trig_series(ch2.time, ch2.value, order)
  mean_frequency = 0.5 * (fit_vx["frequency_hz"] + fit_vy["frequency_hz"])
  omega = 2.0 * np.pi * mean_frequency

  time_grid = build_time_grid(ch1.time)
  vx_fit = evaluate_trig_series(time_grid, fit_vx["coeffs"], omega)
  vy_fit = evaluate_trig_series(time_grid, fit_vy["coeffs"], omega)
  dvy_dt = evaluate_trig_series_derivative(time_grid, fit_vy["coeffs"], omega)

  resistance = float(config["instrument"]["series_resistance_ohm"])
  turns = float(config["instrument"]["coil_turns"])
  coil_length = float(config["instrument"]["coil_length_m"])

  current_fit = vx_fit / resistance
  h_fit = turns * vx_fit / (resistance * coil_length)
  b_fit = 0.5 * vy_fit
  db_dt = 0.5 * dvy_dt
  if blank_slope is not None and meta["specimen_key"] != config["blank"]["specimen_key"]:
    b_fit = b_fit - blank_slope * h_fit
    dh_dt = turns * evaluate_trig_series_derivative(time_grid, fit_vx["coeffs"], omega) / (
      resistance * coil_length
    )
    db_dt = db_dt - blank_slope * dh_dt

  metrics = compute_loop_metrics(h_fit, b_fit)
  param_area, param_signed_area = compute_parametric_loop_area(h_fit, db_dt, time_grid)
  area_display = polygon_area(vx_fit, vy_fit)
  power_loss = param_area * mean_frequency
  metrics["energy_loss_per_cycle_per_volume"] = float(param_area)

  return {
    "specimen": meta["specimen"],
    "specimen_key": meta["specimen_key"],
    "drive_label": meta["drive_label"],
    "run_index": meta["run_index"],
    "frequency_hz": float(mean_frequency),
    "omega_rad_s": float(omega),
    "vx_fit_r2": float(fit_vx["r2"]),
    "vy_fit_r2": float(fit_vy["r2"]),
    "vx_equation": trig_equation_string(fit_vx["coeffs"], omega, "V_x"),
    "vy_equation": trig_equation_string(fit_vy["coeffs"], omega, "V_y"),
    "display_area_v2": float(area_display),
    "current_peak_A": float(np.max(np.abs(current_fit))),
    "blank_corrected": bool(
      blank_slope is not None and meta["specimen_key"] != config["blank"]["specimen_key"]
    ),
    "blank_slope_T_per_Am": blank_slope,
    "power_loss_density_W_per_m3": float(power_loss),
    "files_CH1": ch1.file_name,
    "files_CH2": ch2.file_name,
    "loop_area_parametric_J_per_m3": float(param_area),
    "signed_loop_area_parametric_J_per_m3": float(param_signed_area),
    **metrics,
    "time_grid_s": time_grid.tolist(),
    "vx_fit_V": vx_fit.tolist(),
    "vy_fit_V": vy_fit.tolist(),
    "H_fit_Am": h_fit.tolist(),
    "B_fit_T": b_fit.tolist(),
  }


def plot_time_fits(results: list[dict[str, Any]], out_dir: Path) -> None:
  for result in results:
    time = np.array(result["time_grid_s"])
    vx = np.array(result["vx_fit_V"])
    vy = np.array(result["vy_fit_V"])

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    axes[0].plot(time, vx, color="black", linewidth=1.7)
    axes[0].set_ylabel(r"$V_x~[\mathrm{V}]$")
    axes[0].set_title(f"{result['specimen']} ({result['drive_label']}): harmonic fit of current-channel voltage")
    axes[0].text(
      0.02,
      0.98,
      result["vx_equation"],
      transform=axes[0].transAxes,
      va="top",
      ha="left",
      fontsize=8,
      bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.6"},
    )

    axes[1].plot(time, vy, color="black", linewidth=1.7)
    axes[1].set_ylabel(r"$V_y~[\mathrm{V}]$")
    axes[1].set_xlabel(r"$t~[\mathrm{s}]$")
    axes[1].set_title("Harmonic fit of probe voltage")
    axes[1].text(
      0.02,
      0.98,
      result["vy_equation"],
      transform=axes[1].transAxes,
      va="top",
      ha="left",
      fontsize=8,
      bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.6"},
    )
    fig.savefig(out_dir / f"fit_{result['specimen_key']}_{result['drive_label']}.png")
    plt.close(fig)


def plot_bh_loops(results: list[dict[str, Any]], out_dir: Path) -> None:
  specimen_keys = list(dict.fromkeys(result["specimen_key"] for result in results))
  for specimen_key in specimen_keys:
    group = [result for result in results if result["specimen_key"] == specimen_key]
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    eq_ax = fig.add_subplot(gs[0, 1])

    for result in group:
      h = np.array(result["H_fit_Am"])
      b = np.array(result["B_fit_T"])
      label = (
        rf"{result['drive_label']}: "
        rf"$A_{{\mathrm{{loop}}}}={result['loop_area_parametric_J_per_m3']:.1f}\,\mathrm{{J\,m^{{-3}}}}$, "
        rf"$E_{{\mathrm{{loss}}}}={result['loop_area_parametric_J_per_m3']:.1f}\,\mathrm{{J\,m^{{-3}}\,cycle^{{-1}}}}$"
      )
      ax.plot(h, b, linewidth=2.2, label=label)

    ax.set_title(f"{group[0]['specimen']}: $B$-$H$ loops from harmonic fitting")
    ax.set_xlabel(r"Magnetic field, $H~[\mathrm{A\,m^{-1}}]$")
    ax.set_ylabel(r"Flux density, $B~[\mathrm{T}]$")
    ax.axhline(0.0, color="0.78", linewidth=0.9)
    ax.axvline(0.0, color="0.78", linewidth=0.9)
    ax.legend(loc="upper left", framealpha=0.95)

    eq_ax.axis("off")
    eq_ax.set_title("Fitted Equations", pad=10)
    eq_text = "\n\n".join(
      [
        "\n".join(
          [
            rf"$\mathbf{{{result['drive_label']}}}$",
            result["vx_equation"],
            result["vy_equation"],
          ]
        )
        for result in group
      ]
    )
    eq_ax.text(
      0.02,
      0.98,
      eq_text,
      transform=eq_ax.transAxes,
      va="top",
      ha="left",
      fontsize=9,
      bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.95, "edgecolor": "0.6"},
    )
    fig.savefig(out_dir / f"bh_loop_{specimen_key}.png")
    plt.close(fig)


def plot_comparison(results: list[dict[str, Any]], out_dir: Path) -> None:
  highest_drive = [result for result in results if result["drive_label"] == "C-V3"]
  if not highest_drive:
    return
  fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
  for result in highest_drive:
    h = np.array(result["H_fit_Am"])
    b = np.array(result["B_fit_T"])
    ax.plot(h, b, linewidth=2.1, label=result["specimen"])
  ax.set_title(r"Comparison of $B$-$H$ loops at highest drive ($C$-$V3$)")
  ax.set_xlabel(r"Magnetic field, $H~[\mathrm{A\,m^{-1}}]$")
  ax.set_ylabel(r"Flux density, $B~[\mathrm{T}]$")
  ax.legend(framealpha=0.95)
  fig.savefig(out_dir / "bh_comparison_v3.png")
  plt.close(fig)


def write_csv(results: list[dict[str, Any]], out_path: Path) -> None:
  rows = []
  for result in results:
    row = {key: value for key, value in result.items() if not isinstance(value, list)}
    rows.append(row)
  pd.DataFrame(rows).to_csv(out_path, index=False)


def format_table_value(value: float, digits: int = 3) -> str:
  if abs(value) >= 1e4 or (abs(value) > 0 and abs(value) < 1e-3):
    return f"{value:.3e}"
  return f"{value:.{digits}f}"


def write_tables(results: list[dict[str, Any]], final_dir: Path) -> None:
  columns = [
    "specimen",
    "drive_label",
    "frequency_hz",
    "loop_area_bh",
    "loop_area_parametric_J_per_m3",
    "energy_loss_per_cycle_per_volume",
    "power_loss_density_W_per_m3",
    "remanence_B",
    "coercivity_H",
    "initial_slope_dBdH",
    "vx_fit_r2",
    "vy_fit_r2",
  ]
  df = pd.DataFrame(results)[columns].copy()
  df = df.rename(
    columns={
      "specimen": "Specimen",
      "drive_label": "Drive",
      "frequency_hz": "f_Hz",
      "loop_area_bh": "LoopAreaShoelace_J_per_m3",
      "loop_area_parametric_J_per_m3": "LoopAreaParametric_J_per_m3",
      "energy_loss_per_cycle_per_volume": "EnergyLossPerCycle_J_per_m3",
      "power_loss_density_W_per_m3": "PowerLoss_W_per_m3",
      "remanence_B": "Br_T",
      "coercivity_H": "Hc_A_per_m",
      "initial_slope_dBdH": "dBdH_0",
      "vx_fit_r2": "R2_Vx",
      "vy_fit_r2": "R2_Vy",
    }
  )
  df.to_csv(final_dir / "important_results_table.csv", index=False)

  comparison = (
    df[df["Drive"] == "C-V3"]
    .sort_values("EnergyLossPerCycle_J_per_m3", ascending=False)
    .reset_index(drop=True)
  )
  comparison.to_csv(final_dir / "comparison_table_v3.csv", index=False)

  md_lines = []
  md_lines.append(r"$A_{\mathrm{loop}} = \left|\int_0^T H(t)\,\frac{dB}{dt}\,dt\right|,\quad E_{\mathrm{loss}} = A_{\mathrm{loop}},\quad P_{\mathrm{loss}} = fE_{\mathrm{loss}}$")
  md_lines.append("")
  md_lines.append("| Specimen | Drive | f (Hz) | Parametric loop area $A_{loop}$ (J/m^3) | Energy loss per cycle (J/m^3) | Power loss (W/m^3) | $B_r$ (T) | $H_c$ (A/m) |")
  md_lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
  for _, row in df.iterrows():
    md_lines.append(
      f"| {row['Specimen']} | {row['Drive']} | {row['f_Hz']:.3f} | "
      f"{row['LoopAreaParametric_J_per_m3']:.3f} | {row['EnergyLossPerCycle_J_per_m3']:.3f} | "
      f"{row['PowerLoss_W_per_m3']:.3f} | {row['Br_T']:.5f} | {row['Hc_A_per_m']:.3f} |"
    )
  (final_dir / "important_results_table.md").write_text("\n".join(md_lines), encoding="utf-8")


def write_latex_table(results: list[dict[str, Any]], final_dir: Path) -> Path:
  df = pd.DataFrame(results)[
    [
      "specimen",
      "drive_label",
      "frequency_hz",
      "loop_area_parametric_J_per_m3",
      "energy_loss_per_cycle_per_volume",
      "power_loss_density_W_per_m3",
      "remanence_B",
      "coercivity_H",
    ]
  ].copy()

  rows = []
  for _, row in df.iterrows():
    rows.append(
      " {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
        row["specimen"],
        row["drive_label"],
        format_table_value(row["frequency_hz"], 2),
        format_table_value(row["loop_area_parametric_J_per_m3"], 2),
        format_table_value(row["energy_loss_per_cycle_per_volume"], 2),
        format_table_value(row["power_loss_density_W_per_m3"], 2),
        format_table_value(row["remanence_B"], 5),
        format_table_value(row["coercivity_H"], 2),
      )
    )

  tex = "\n".join(
    [
      r"\documentclass[varwidth=24cm,border=8pt]{standalone}",
      r"\usepackage{booktabs}",
      r"\usepackage{amsmath}",
      r"\begin{document}",
      r"\begin{minipage}{23cm}",
      r"\centering",
      r"{\Large \textbf{Complex-Fit B-H Curve Results}}\\[0.4em]",
      r"{\small Harmonic fitting model for both channels: }\\[0.2em]",
      r"{\small $V(t)=a_0+\sum_{n=1}^{N}\left[a_n\sin(n\omega t)+b_n\cos(n\omega t)\right],\quad N=5$}\\[0.5em]",
      r"{\small $A_{\mathrm{loop}}=\left|\int_0^T H(t)\,\dfrac{dB}{dt}\,dt\right|,\quad E_{\mathrm{loss}}=A_{\mathrm{loop}},\quad P_{\mathrm{loss}}=fE_{\mathrm{loss}}$}\\[0.7em]",
      r"\renewcommand{\arraystretch}{1.2}",
      r"\begin{tabular}{llrrrrrr}",
      r"\toprule",
      r"Specimen & Drive & $f~(\mathrm{Hz})$ & $A_{\mathrm{loop}}~(\mathrm{J\,m^{-3}})$ & $E_{\mathrm{loss}}~(\mathrm{J\,m^{-3}\,cycle^{-1}})$ & $P_{\mathrm{loss}}~(\mathrm{W\,m^{-3}})$ & $B_r~(\mathrm{T})$ & $H_c~(\mathrm{A\,m^{-1}})$ \\",
      r"\midrule",
      *rows,
      r"\bottomrule",
      r"\end{tabular}",
      r"\end{minipage}",
      r"\end{document}",
    ]
  )
  tex_path = final_dir / "important_results_table.tex"
  tex_path.write_text(tex, encoding="utf-8")
  return tex_path


def render_latex_table(tex_path: Path) -> None:
  pdf_path = tex_path.with_suffix(".pdf")
  png_path = tex_path.with_suffix(".png")
  subprocess.run(
    [
      "pdflatex",
      "-interaction=nonstopmode",
      "-output-directory",
      str(tex_path.parent),
      str(tex_path),
    ],
    check=True,
    capture_output=True,
  )

  converter_commands = [
    ["pdftocairo", "-png", "-singlefile", "-r", "300", str(pdf_path), str(png_path.with_suffix(""))],
    ["magick", "-density", "300", str(pdf_path), str(png_path)],
  ]
  for command in converter_commands:
    try:
      subprocess.run(command, check=True, capture_output=True)
      if png_path.exists():
        return
    except Exception:
      continue


def write_summary(results: list[dict[str, Any]], blank_slope: float, out_path: Path, config: dict[str, Any]) -> None:
  lines = []
  lines.append("B-H CURVE ANALYSIS REPORT (COMPLEX FIT)")
  lines.append("=" * 40)
  lines.append("")
  lines.append("Model and Formulas")
  lines.append("-" * 18)
  lines.append(
    "Both oscilloscope channels were fitted with a harmonic trigonometric model "
    "V(t) = a0 + sum[a_n sin(nwt) + b_n cos(nwt)], using harmonics up to N = "
    f"{config['fit']['harmonic_order']}."
  )
  lines.append(
    f"H = N*Vx/(R*L), with N={config['instrument']['coil_turns']}, "
    f"R={config['instrument']['series_resistance_ohm']} ohm, "
    f"L={config['instrument']['coil_length_m']} m"
  )
  lines.append("B = 0.5*Vy [tesla]")
  lines.append("Parametric loop-area formula: A_loop = |integral H(t) * (dB/dt) dt over one cycle|")
  lines.append("Energy loss per cycle per unit volume: E_loss = A_loop")
  lines.append("Power loss per unit volume: P_loss = f * E_loss")
  lines.append("")
  lines.append("Blank correction: B_corrected = B_measured - alpha*H")
  lines.append(f"alpha = {blank_slope:.6e} T m/A")
  lines.append("")

  grouped: dict[str, list[dict[str, Any]]] = {}
  for result in results:
    grouped.setdefault(result["specimen"], []).append(result)

  for specimen, group in grouped.items():
    lines.append(specimen)
    lines.append("-" * len(specimen))
    for result in group:
      lines.append(
        f"{result['drive_label']}: "
        f"f={result['frequency_hz']:.3f} Hz, "
        f"R2(Vx)={result['vx_fit_r2']:.5f}, "
        f"R2(Vy)={result['vy_fit_r2']:.5f}, "
        f"A_loop={result['loop_area_parametric_J_per_m3']:.6g} J/m^3, "
        f"E_loss={result['energy_loss_per_cycle_per_volume']:.6g} J/m^3/cycle, "
        f"P_loss={result['power_loss_density_W_per_m3']:.6g} W/m^3, "
        f"Br={result['remanence_B']:.6g} T, "
        f"Hc={result['coercivity_H']:.6g} A/m"
      )
    lines.append("")

  v3_runs = [result for result in results if result["drive_label"] == "C-V3"]
  if v3_runs:
    ranked = sorted(v3_runs, key=lambda item: item["energy_loss_per_cycle_per_volume"], reverse=True)
    lines.append("Energy-loss ranking at C-V3: " + " > ".join(item["specimen"] for item in ranked))

  out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
  paths = get_paths()
  config = load_config(paths["config"])
  pairs = pair_traces(paths["raw"])
  if len(pairs) != len(config["runs"]):
    raise ValueError("The number of trace pairs does not match the configured run list.")

  blank_index = int(config["blank"]["run_index"])
  blank_result = analyze_pair(pairs[blank_index], config["runs"][blank_index], config, None)
  h_blank = np.array(blank_result["H_fit_Am"])
  b_blank = 0.5 * np.array(blank_result["vy_fit_V"])
  blank_slope, _ = np.polyfit(h_blank, b_blank, 1)

  results = []
  for meta, pair in zip(config["runs"], pairs):
    result = analyze_pair(pair, meta, config, float(blank_slope))
    if meta["specimen_key"] != config["blank"]["specimen_key"]:
      results.append(result)

  plot_time_fits(results, paths["plots"])
  plot_bh_loops(results, paths["plots"])
  plot_comparison(results, paths["plots"])
  write_csv(results, paths["final"] / "bh_analysis_results.csv")
  write_tables(results, paths["final"])
  tex_path = write_latex_table(results, paths["final"])
  render_latex_table(tex_path)
  write_summary(results, float(blank_slope), paths["final"] / "analysis_summary.txt", config)


if __name__ == "__main__":
  main()
