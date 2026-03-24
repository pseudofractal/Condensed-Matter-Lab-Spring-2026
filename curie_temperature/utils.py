import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

def load_data(filepath):
    df = pd.read_csv(filepath)

    # Remove obvious typo outliers (e.g., 3.9 V mistake)
    df = df[df["Vdc_V"] < 3.5]

    # Average duplicate temperatures
    df = df.groupby("Temperature_C", as_index=False).mean()

    return df


def apply_vtotal_method(df):
    """
    Apply required method:
    1. Compute V_total_avg from rows where both Vdc and Vsc exist
    2. For missing Vsc:
       Vsc = V_total_avg - Vdc
    """

    valid = df.dropna(subset=["Vdc_V", "Vsc_V"]).copy()
    if valid.empty:
        raise ValueError("No rows contain both Vdc_V and Vsc_V for V_total fitting.")

    valid["V_total"] = valid["Vdc_V"] + valid["Vsc_V"]
    valid = valid.sort_values("Temperature_C").reset_index(drop=True)

    x_fit = valid["Temperature_C"].to_numpy(dtype=float)
    y_fit = valid["V_total"].to_numpy(dtype=float)

    # Fit constant model y = a using only rows where both voltages exist.
    def constant_model(x, a):
        return np.full_like(x, a, dtype=float)

    popt, _ = curve_fit(constant_model, x_fit, y_fit, p0=[np.mean(y_fit)])
    V_total_fit = float(popt[0])

    df["Vsc_V"] = df.apply(
        lambda row: V_total_fit - row["Vdc_V"]
        if pd.isna(row["Vsc_V"]) else row["Vsc_V"],
        axis=1
    )

    return df, V_total_fit, valid


def compute_capacitance(df):
    df["C"] = df["Vsc_V"] / df["Vdc_V"]

    # Reference capacitance below 60°C
    C_ref = df[df["Temperature_C"] < 60]["C"].mean()

    df["epsilon_r"] = df["C"] / C_ref

    return df, C_ref


def _sanitize_savgol_window(n_points, smooth_window, smooth_polyorder):
    # Ensure an odd, valid Savitzky-Golay window length.
    window = min(smooth_window, n_points if n_points % 2 == 1 else n_points - 1)
    if window < 3:
        window = 3
    if window <= smooth_polyorder:
        window = smooth_polyorder + 1
        if window % 2 == 0:
            window += 1
    if window > n_points:
        window = n_points if n_points % 2 == 1 else n_points - 1
    return window


def _estimate_transition_core(temperature, epsilon_r, smooth_window, smooth_polyorder):
    n_points = len(epsilon_r)
    if n_points < 3:
        raise ValueError("Need at least 3 points to estimate transition temperature.")

    window = _sanitize_savgol_window(n_points, smooth_window, smooth_polyorder)

    if window >= 3 and window > smooth_polyorder:
        eps_smooth = savgol_filter(epsilon_r, window_length=window, polyorder=smooth_polyorder)
    else:
        eps_smooth = epsilon_r.copy()

    d_eps_dT = np.gradient(eps_smooth, temperature)

    peak_idx = int(np.argmax(eps_smooth))
    deriv_idx = int(np.argmax(d_eps_dT))
    return {
        "eps_smooth": eps_smooth,
        "d_eps_dT": d_eps_dT,
        "peak_idx": peak_idx,
        "deriv_idx": deriv_idx,
        "Tc_peak": float(temperature[peak_idx]),
        "Tc_derivative": float(temperature[deriv_idx]),
        "epsilon_at_peak": float(eps_smooth[peak_idx]),
        "d_eps_dT_at_peak": float(d_eps_dT[deriv_idx]),
    }


def compute_transition_temperature(
    df,
    smooth_window=9,
    smooth_polyorder=2,
    temp_least_count_c=0.1,
    voltage_least_count_v=0.1,
    mc_samples=400,
):
    df = df.sort_values("Temperature_C").reset_index(drop=True).copy()

    temperature = df["Temperature_C"].to_numpy(dtype=float)
    epsilon_r = df["epsilon_r"].to_numpy(dtype=float)

    n_points = len(epsilon_r)
    if n_points < 3:
        raise ValueError("Need at least 3 points to estimate transition temperature.")

    core = _estimate_transition_core(
        temperature=temperature,
        epsilon_r=epsilon_r,
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
    )
    eps_smooth = core["eps_smooth"]
    d_eps_dT = core["d_eps_dT"]
    peak_idx = core["peak_idx"]
    deriv_idx = core["deriv_idx"]
    tc_peak = core["Tc_peak"]
    tc_derivative = core["Tc_derivative"]

    def local_temp_unc(i):
        left = abs(temperature[i] - temperature[i - 1]) if i > 0 else np.inf
        right = abs(temperature[i + 1] - temperature[i]) if i < n_points - 1 else np.inf
        step = min(left, right)
        return 0.5 * step if np.isfinite(step) else 0.0

    # Temperature resolution contribution from instrument least count (uniform quantization).
    temp_quant_std = float(temp_least_count_c) / np.sqrt(12.0)
    tc_peak_local_unc = float(local_temp_unc(peak_idx))
    tc_derivative_local_unc = float(local_temp_unc(deriv_idx))

    # Voltage least-count propagation to epsilon_r uncertainty.
    u_v = float(voltage_least_count_v) / np.sqrt(12.0)
    vdc = df["Vdc_V"].to_numpy(dtype=float)
    vsc = df["Vsc_V"].to_numpy(dtype=float)
    C = df["C"].to_numpy(dtype=float)
    c_ref_mask = df["Temperature_C"].to_numpy(dtype=float) < 60.0
    c_ref_vals = C[c_ref_mask]
    if len(c_ref_vals) == 0:
        c_ref_vals = C
    C_ref = float(np.mean(c_ref_vals))

    with np.errstate(divide="ignore", invalid="ignore"):
        uC = np.abs(C) * np.sqrt((u_v / np.abs(vsc)) ** 2 + (u_v / np.abs(vdc)) ** 2)
    uC = np.where(np.isfinite(uC), uC, 0.0)

    uC_ref = float(np.sqrt(np.sum(uC[c_ref_mask] ** 2)) / max(1, np.count_nonzero(c_ref_mask)))
    if not np.isfinite(uC_ref) or uC_ref <= 0.0:
        uC_ref = float(np.sqrt(np.sum(uC ** 2)) / max(1, len(uC)))

    with np.errstate(divide="ignore", invalid="ignore"):
        u_eps = np.abs(epsilon_r) * np.sqrt((uC / np.abs(C)) ** 2 + (uC_ref / abs(C_ref)) ** 2)
    u_eps = np.where(np.isfinite(u_eps), u_eps, 0.0)

    # Monte Carlo for Tc sensitivity to measurement uncertainty.
    rng = np.random.default_rng(2026)
    tc_peak_mc = []
    tc_deriv_mc = []
    n_mc = max(0, int(mc_samples))
    for _ in range(n_mc):
        temp_jitter = rng.uniform(-0.5 * temp_least_count_c, 0.5 * temp_least_count_c, size=n_points)
        eps_jitter = rng.normal(0.0, u_eps, size=n_points)
        t_sample = temperature + temp_jitter
        e_sample = epsilon_r + eps_jitter
        order = np.argsort(t_sample)
        t_sorted = t_sample[order]
        e_sorted = e_sample[order]
        try:
            sample_core = _estimate_transition_core(
                temperature=t_sorted,
                epsilon_r=e_sorted,
                smooth_window=smooth_window,
                smooth_polyorder=smooth_polyorder,
            )
            tc_peak_mc.append(sample_core["Tc_peak"])
            tc_deriv_mc.append(sample_core["Tc_derivative"])
        except ValueError:
            continue

    tc_peak_mc_std = float(np.std(tc_peak_mc, ddof=1)) if len(tc_peak_mc) > 1 else 0.0
    tc_derivative_mc_std = float(np.std(tc_deriv_mc, ddof=1)) if len(tc_deriv_mc) > 1 else 0.0

    tc_peak_unc = float(np.sqrt(tc_peak_local_unc ** 2 + temp_quant_std ** 2 + tc_peak_mc_std ** 2))
    tc_derivative_unc = float(
        np.sqrt(tc_derivative_local_unc ** 2 + temp_quant_std ** 2 + tc_derivative_mc_std ** 2)
    )

    # Method-to-method spread is treated as an additional systematic component.
    method_difference = abs(tc_peak - tc_derivative)
    tc_combined = 0.5 * (tc_peak + tc_derivative)
    avg_local_unc = 0.5 * (tc_peak_unc + tc_derivative_unc)
    tc_combined_unc = float(np.sqrt((0.5 * method_difference) ** 2 + avg_local_unc ** 2))
    tc_combined_unc_pct = (
        float((tc_combined_unc / abs(tc_combined)) * 100.0) if tc_combined != 0 else np.nan
    )

    residual = epsilon_r - eps_smooth
    epsilon_noise_std = float(np.std(residual, ddof=1)) if n_points > 1 else 0.0

    return {
        "temperature": temperature,
        "epsilon_smooth": eps_smooth,
        "d_eps_dT": d_eps_dT,
        "Tc_peak": tc_peak,
        "Tc_peak_unc": tc_peak_unc,
        "Tc_derivative": tc_derivative,
        "Tc_derivative_unc": tc_derivative_unc,
        "Tc_combined": float(tc_combined),
        "Tc_combined_unc": tc_combined_unc,
        "Tc_combined_unc_pct": tc_combined_unc_pct,
        "Tc_method_difference": float(method_difference),
        "epsilon_at_peak": core["epsilon_at_peak"],
        "d_eps_dT_at_peak": core["d_eps_dT_at_peak"],
        "epsilon_noise_std": epsilon_noise_std,
        "epsilon_instr_unc_mean": float(np.mean(u_eps)),
        "Tc_peak_mc_std": tc_peak_mc_std,
        "Tc_derivative_mc_std": tc_derivative_mc_std,
    }
