"""
Generate new calibration curve plot for new silver data.
Uses Mathtext (internal engine) instead of external LaTeX.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def format_scientific_mathtext(value, include_sign=True):
  """
  Format a number in scientific notation for Mathtext.
  Returns format like: 3.45 \times 10^{-2}
  """
  if abs(value) < 1e-15:
    return "0"

  sign_str = "-" if value < 0 else ""
  if not include_sign:
    sign_str = ""

  abs_val = abs(value)
  exponent = int(np.floor(np.log10(abs_val)))
  mantissa = abs_val / (10**exponent)

  # Format with up to 3 decimal places, strip trailing zeros
  mantissa_str = f"{mantissa:.3f}".rstrip("0").rstrip(".")

  return f"{sign_str}{mantissa_str} \\times 10^{{{exponent}}}"


def format_polynomial_mathtext(coeffs, max_terms_per_line=3):
  """
  Format polynomial coefficients using Mathtext syntax.
  """
  terms = []
  degree = len(coeffs) - 1

  for i, coef in enumerate(coeffs):
    power = degree - i
    if abs(coef) < 1e-15:
      continue

    # Handle sign logic for cleaner joining
    is_first = len(terms) == 0

    if is_first:
      term_str = format_scientific_mathtext(coef, include_sign=True)
      prefix = ""
    elif coef < 0:
      prefix = " - "
      term_str = format_scientific_mathtext(coef, include_sign=False)
    else:
      prefix = " + "
      term_str = format_scientific_mathtext(coef, include_sign=True)

    if power == 0:
      core = f"{term_str}"
    elif power == 1:
      core = f"{term_str}x"
    else:
      core = f"{term_str}x^{{{power}}}"

    terms.append(f"{prefix}{core}")

  if not terms:
    return "0"

  # Join all terms into one long string
  # Mathtext does not support multiline arrays easily, so we output a single line.
  # If it is too long, we can manually break it with a newline '\n' but it won't align perfectly.
  full_eq = "".join(terms)

  # If you really need a split, you can do a rough split here:
  midpoint = len(full_eq) // 2
  if len(full_eq) > 60:  # Arbitrary length limit
    # Find a convenient plus or minus near the middle to break on
    split_idx = full_eq.find(" + ", midpoint)
    if split_idx == -1:
      split_idx = full_eq.find(" - ", midpoint)

    if split_idx != -1:
      part1 = full_eq[:split_idx]
      part2 = full_eq[split_idx:]
      return f"{part1}\n{part2}"

  return full_eq


# --- MATPLOTLIB CONFIGURATION ---
plt.rcParams.update(
  {
    "text.usetex": False,  # Disable external LaTeX requirement
    "font.family": "serif",  # Use serif font for normal text
    "mathtext.fontset": "cm",  # Use 'Computer Modern' (TeX-like) for math
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "figure.dpi": 300,
  }
)


def main():
  parser = argparse.ArgumentParser(description="Generate calibration curve")
  parser.add_argument("--degree", type=int, default=6)
  args = parser.parse_args()

  # Determine paths safely
  try:
    base_dir = Path(__file__).parent
  except NameError:
    base_dir = Path(".")

  data_path = base_dir / "data" / "new_silver_data.csv"
  plots_dir = base_dir / "plots"
  plots_dir.mkdir(parents=True, exist_ok=True)
  output_path = plots_dir / "new_calibration_curve.png"

  # Load Data (or dummy data if file missing)
  if not data_path.exists():
    print("Data file not found, creating dummy data for demo...")
    temps = np.linspace(22, 35, 20)
    intervals = 2 * temps**2 - 50 * temps + 400 + np.random.normal(0, 5, 20)
  else:
    df = pd.read_csv(data_path)
    temps = df["Temp_C"].to_numpy()
    intervals = df["dt_s"].to_numpy()

  # Plot Setup
  fig, ax = plt.subplots(figsize=(8, 6))

  ax.plot(temps, intervals, "o", color="blue", label="Experimental Data")

  # Polynomial Fit
  z = np.polyfit(temps, intervals, args.degree)
  p = np.poly1d(z)

  x_smooth = np.linspace(temps.min(), temps.max(), 200)
  ax.plot(x_smooth, p(x_smooth), "r--", label=f"Fit (deg {args.degree})")

  # Equation String
  # Note: We wrap it in $...$ for Mathtext rendering
  eq_content = format_polynomial_mathtext(z)
  equation_text = f"$y = {eq_content}$"

  # Add Text Box
  # Using 'multialignment' handles the newline if we split the equation
  ax.text(
    0.98,
    0.05,
    equation_text,
    transform=ax.transAxes,
    verticalalignment="bottom",
    horizontalalignment="right",
    multialignment="right",
    fontsize=8,
    bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9),
  )

  # Info Box
  ax.text(
    0.05,
    0.95,
    "Material: Silver\nCurrent: 284 mA",
    transform=ax.transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", fc="white", alpha=0.85),
  )

  # Labels using raw strings r"" for math symbols
  ax.set_xlabel(r"Temperature $T$ ($^\circ$C)")
  ax.set_ylabel(r"Time Interval $\Delta t$ (s)")
  ax.set_title(r"Calibration Curve: $Ag$ Data")
  ax.legend()

  plt.tight_layout()
  plt.savefig(output_path)
  plt.close()
  print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
  main()
