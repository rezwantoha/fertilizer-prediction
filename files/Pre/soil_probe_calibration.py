"""
soil_probe_calibration.py
=========================
Deployable calibration layer for the 7-in-1 RS485/Modbus soil probe.

Fitted on the 18-sample spike-recovery experiment (Fertilization_Comparison.xlsx),
16 samples after outlier exclusion (K5, P1). See calibration_report.md for the
full derivation and validation.

WHAT THIS MODULE DOES
    - collapses the redundant N/P/K channels into ONE physically meaningful
      quantity: composite ionic loading (CIL), in mg/kg K-equivalent
    - maps CIL onto a 3-level ordinal soil-test class for FRG-2018 lookup
    - applies a single-point pH offset

WHAT THIS MODULE DELIBERATELY DOES *NOT* DO
    - it does not return separate N, P and K concentrations, because the probe
      cannot measure them separately (det of the sensitivity matrix ~ 1.7e-8,
      condition number ~ 3.1e4, effective rank 1)

Usage
-----
    from soil_probe_calibration import calibrate

    r = calibrate(ph_raw=7.9, n_pct=0.006, p_raw=177.0, k_raw=0.435, rh=52.2)
    print(r["cil_mgkg_Keq"], r["cil_ci95"], r["class_label"])
"""

from __future__ import annotations
from dataclasses import dataclass, asdict

# ---------------------------------------------------------------------------
# Fitted constants
# ---------------------------------------------------------------------------

# Layer 0 -- firmware redundancy check.  K_raw is an exact affine map of P_raw.
K_FROM_P_SLOPE = 0.00257856
K_FROM_P_INTERCEPT = -0.02064253
K_FROM_P_R2 = 0.999969
K_FROM_P_RESID_SD = 0.001215          # raw units; 3-sigma gate = 0.00365

# Layer 1 -- forward model  P_raw = b0 + bN*N + bP*P + bK*K   (mg/kg applied)
FWD_B0, FWD_BN, FWD_BP, FWD_BK = 74.0253, 0.725386, 0.184483, 1.234591
FWD_SE = (5.8596, 0.081343, 0.133643, 0.077431)
FWD_R2, FWD_RMSE = 0.97058, 14.804
# NOTE: bP is NOT statistically significant (t = 1.38, p = 0.19). Phosphorus
#       is invisible to this probe within the tested 15-75 mg/kg range.

# Layer 2 -- composite ionic loading weights (K-equivalent basis)
W_N, W_P, W_K = 0.587551, 0.149428, 1.0

# Layer 3 -- deployable inverse:  CIL = a * P_raw + b
CIL_SLOPE = 0.78615442
CIL_INTERCEPT = -56.40418776
CIL_R2 = 0.97058
CIL_RMSE = 10.937                     # mg/kg K-eq, 1 sigma
CIL_LOD = 32.81                       # 3 sigma
CIL_LOQ = 109.37                      # 10 sigma
CIL_T95 = 2.145                       # t(0.975, dof=14)

# Layer 4 -- ordinal class edges (mg/kg K-equivalent)
CLASS_EDGES = (29.38, 60.00)
CLASS_LABELS = ("Low", "Medium", "High")
CLASS_ACC_LOO = 0.938
CLASS_KAPPA_LOO = 0.906

# Layer 5 -- pH
PH_OFFSET = 1.70                      # single-point, from baseline soil only
PH_REPEATABILITY_SD = 0.45            # across nominally identical samples

# Validity envelope of the calibration experiment
P_RAW_MIN, P_RAW_MAX = 71.0, 351.8
RH_MIN, RH_MAX = 49.3, 62.8
T_MIN, T_MAX = 27.8, 28.8


@dataclass
class CalibrationResult:
    cil_mgkg_Keq: float
    cil_ci95: tuple[float, float]
    class_index: int
    class_label: str
    ph_calibrated: float | None
    ph_uncertainty: float
    flags: list[str]
    quantitative_npk: None = None      # intentionally always None

    def as_dict(self):
        return asdict(self)


def composite_index(n_mgkg: float, p_mgkg: float, k_mgkg: float) -> float:
    """Forward direction: convert a known N-P-K loading into K-equivalent CIL.

    Used to build the training target, and to translate an FRG-2018
    recommendation back onto the axis the probe can actually observe.
    """
    return W_N * n_mgkg + W_P * p_mgkg + W_K * k_mgkg


def predicted_raw(n_mgkg: float, p_mgkg: float, k_mgkg: float) -> float:
    """Forward model: what the P channel should read for a given loading."""
    return FWD_B0 + FWD_BN * n_mgkg + FWD_BP * p_mgkg + FWD_BK * k_mgkg


def calibrate(ph_raw: float,
              n_pct: float,
              p_raw: float,
              k_raw: float,
              rh: float | None = None,
              temp_c: float | None = None,
              apply_ph_offset: bool = False) -> CalibrationResult:
    """Apply the full calibration stack to one probe reading.

    Parameters
    ----------
    ph_raw  : probe pH channel
    n_pct   : probe nitrogen channel, %
    p_raw   : probe phosphorus channel, raw counts  <- the only real measurement
    k_raw   : probe potassium channel, raw counts   <- redundant, used as a check
    rh, temp_c : moisture (%) and temperature (C), used only for validity flags
    apply_ph_offset : apply the provisional single-point pH offset (default False)
    """
    flags: list[str] = []

    # --- Layer 0: verify the probe is behaving as characterised --------------
    k_expected = K_FROM_P_SLOPE * p_raw + K_FROM_P_INTERCEPT
    if abs(k_raw - k_expected) > 3 * K_FROM_P_RESID_SD:
        flags.append(
            f"K/P consistency check failed (expected {k_expected:.3f}, "
            f"got {k_raw:.3f}) - firmware or wiring differs from the "
            "calibrated unit; recalibrate before trusting this reading.")

    # --- Layer 1: N channel is below its own quantisation floor -------------
    if n_pct < 0.010:
        flags.append(
            f"N channel reads {n_pct:.3f} % - within 10 LSB of zero "
            "(LSB = 0.001 % = 10 mg/kg). Treat as non-informative.")

    # --- Layer 2/3: composite ionic loading ---------------------------------
    cil = CIL_SLOPE * p_raw + CIL_INTERCEPT
    half = CIL_T95 * CIL_RMSE
    ci = (cil - half, cil + half)

    if cil < CIL_LOQ:
        flags.append(
            f"CIL {cil:.1f} < LOQ {CIL_LOQ:.0f} mg/kg K-eq - report the ordinal "
            "class, not the number.")

    # --- Layer 4: ordinal class ---------------------------------------------
    idx = 0 if cil < CLASS_EDGES[0] else (1 if cil < CLASS_EDGES[1] else 2)

    # --- Layer 5: pH ---------------------------------------------------------
    # The +1.70 single-point offset is derived from ONE lab pH value and pushes
    # several readings above pH 9.5, which is not physical for this soil. It is
    # therefore NOT applied by default. Do a 2-point buffer calibration
    # (pH 4.01 / 7.00) and replace PH_OFFSET with a proper slope+intercept.
    if apply_ph_offset:
        ph_cal = None if ph_raw is None else ph_raw + PH_OFFSET
        flags.append("Single-point pH offset applied - provisional, not validated.")
    else:
        ph_cal = ph_raw
    if ph_raw is not None and not (3.5 <= ph_raw <= 9.5):
        flags.append("pH channel out of physical range - probe contact fault.")

    # --- validity envelope ---------------------------------------------------
    if not (P_RAW_MIN <= p_raw <= P_RAW_MAX):
        flags.append(
            f"P channel {p_raw:.1f} is outside the calibrated range "
            f"[{P_RAW_MIN}, {P_RAW_MAX}] - extrapolation, uncertainty unbounded.")
    if rh is not None and not (RH_MIN <= rh <= RH_MAX):
        flags.append(
            f"Moisture {rh:.1f} % is outside the calibrated band "
            f"[{RH_MIN}, {RH_MAX}] - EC response is moisture-dependent.")
    if temp_c is not None and not (T_MIN - 3 <= temp_c <= T_MAX + 3):
        flags.append(
            f"Temperature {temp_c:.1f} C is far from the calibration "
            f"temperature (~28 C); EC drifts roughly 2 %/K.")

    return CalibrationResult(
        cil_mgkg_Keq=round(cil, 1),
        cil_ci95=(round(ci[0], 1), round(ci[1], 1)),
        class_index=idx,
        class_label=CLASS_LABELS[idx],
        ph_calibrated=None if ph_cal is None else round(ph_cal, 2),
        ph_uncertainty=PH_REPEATABILITY_SD,
        flags=flags,
    )


if __name__ == "__main__":
    demo = [
        ("K3 (in-range)", dict(ph_raw=6.9, n_pct=0.006, p_raw=177.0, k_raw=0.435, rh=52.2, temp_c=28.4)),
        ("P2 (low end)",  dict(ph_raw=6.9, n_pct=0.001, p_raw=79.0,  k_raw=0.182, rh=60.4, temp_c=28.4)),
        ("M3 (high end)", dict(ph_raw=7.9, n_pct=0.013, p_raw=351.8, k_raw=0.885, rh=51.3, temp_c=28.0)),
        ("bad K channel", dict(ph_raw=7.5, n_pct=0.004, p_raw=150.0, k_raw=0.900, rh=55.0, temp_c=28.0)),
    ]
    for name, kw in demo:
        r = calibrate(**kw)
        print(f"\n{name}")
        print(f"  CIL   = {r.cil_mgkg_Keq} mg/kg K-eq  95% CI {r.cil_ci95}")
        print(f"  class = {r.class_label}   pH = {r.ph_calibrated}")
        for f in r.flags:
            print(f"  [!] {f}")
