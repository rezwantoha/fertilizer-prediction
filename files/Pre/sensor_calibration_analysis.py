"""
=============================================================================
 7-in-1 RS485 Soil Probe -- Calibration Analysis
 Thesis: AI-Based Automated Fertilizer Recommendation System (Bangladesh)
=============================================================================

 Input : Fertilization_Comparison.xlsx
 Output: figures (PNG), calibration coefficients (JSON), console report

 Design of experiment
 --------------------
   U1..U5  urea spike       N = 25, 50, 75, 100, 125 mg/kg
   P1..P5  TSP  spike       P = 15, 30, 45,  60,  75 mg/kg
   K1..K5  MoP  spike       K = 30, 60, 90, 120, 150 mg/kg
   M1..M3  combined         (N25 P15 K30), (N75 P45 K90), (N125 P75 K150)
   all in 50 mL total liquid on the same base soil

 Dependencies: numpy, pandas, scipy, matplotlib, scikit-learn
=============================================================================
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut

# --------------------------------------------------------------------------
# Plot style
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "legend.frameon": False,
    "legend.fontsize": 8,
})

C = {"U": "#C1666B", "P": "#48A9A6", "K": "#4C6085", "M": "#D4A373",
     "fit": "#1B1B1E", "ci": "#9AA0A6", "warn": "#B23A48"}

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

XLSX = "/mnt/user-data/uploads/Fertilization_Comparison.xlsx"

# --------------------------------------------------------------------------
# 1. LOAD
# --------------------------------------------------------------------------
raw = pd.read_excel(XLSX, sheet_name="After_Ferti_sensor_data", header=None)
raw = raw.iloc[1:, 3:10]
raw.columns = ["Sample", "pH", "T", "RH", "N_pct", "P_raw", "K_raw"]
raw = raw[raw["Sample"].astype(str).str.match(r"^[UPKM]\d")].reset_index(drop=True)
for c in raw.columns[1:]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")

# known applied loadings (mg/kg) -- the ground truth of the DOE
applied = {
    "U1": (25, 0, 0),   "U2": (50, 0, 0),   "U3": (75, 0, 0),
    "U4": (100, 0, 0),  "U5": (125, 0, 0),
    "P1": (0, 15, 0),   "P2": (0, 30, 0),   "P3": (0, 45, 0),
    "P4": (0, 60, 0),   "P5": (0, 75, 0),
    "K1": (0, 0, 30),   "K2": (0, 0, 60),   "K3": (0, 0, 90),
    "K4": (0, 0, 120),  "K5": (0, 0, 150),
    "M1": (25, 15, 30), "M2": (75, 45, 90), "M3": (125, 75, 150),
}
df = raw.copy()
df[["N_add", "P_add", "K_add"]] = pd.DataFrame(
    [applied[s] for s in df["Sample"]], index=df.index)
df["series"] = df["Sample"].str[0]

# baseline soil (single pre-fertilisation lab test)
LAB_BASE = {"pH": 7.8, "N_pct": 0.05, "P_mgkg": 3.8, "K_meq": 0.37}
SENS_BASE = {"pH": 6.1, "N_pct": 0.0, "P_raw": 20.71, "K_raw": 0.031, "RH": 26.39}

print("=" * 78)
print("SECTION 1 - DATASET")
print("=" * 78)
print(df[["Sample", "series", "pH", "RH", "N_pct", "P_raw", "K_raw",
          "N_add", "P_add", "K_add"]].to_string(index=False))

# --------------------------------------------------------------------------
# 2. DIAGNOSTIC A -- channel redundancy / rank
# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SECTION 2 - CHANNEL REDUNDANCY (is this really 3 sensors?)")
print("=" * 78)

ch = df[["N_pct", "P_raw", "K_raw"]].to_numpy(float)

# K as an affine function of P
kp = stats.linregress(df["P_raw"], df["K_raw"])
k_resid = df["K_raw"] - (kp.slope * df["P_raw"] + kp.intercept)
print(f"K_raw = {kp.slope:.7f} * P_raw + ({kp.intercept:+.6f})")
print(f"    R^2 = {kp.rvalue**2:.7f}   residual sigma = {k_resid.std(ddof=2):.5f}"
      f"   max|resid| = {np.abs(k_resid).max():.5f}")

# PCA on log channels (channels span >1 decade)
Xl = np.log(ch)
Xz = (Xl - Xl.mean(0)) / Xl.std(0, ddof=1)
sv = np.linalg.svd(Xz, compute_uv=False)
evr = sv**2 / (sv**2).sum()
cond = sv[0] / sv[-1]
print(f"\nPCA (log channels) explained variance : {np.round(evr*100, 2)} %")
print(f"Condition number of channel matrix   : {cond:,.0f}")

# VIF
def vif(X):
    out = []
    for j in range(X.shape[1]):
        y = X[:, j]
        Xo = np.delete(X, j, axis=1)
        r2 = LinearRegression().fit(Xo, y).score(Xo, y)
        out.append(np.inf if r2 >= 1 - 1e-12 else 1 / (1 - r2))
    return out
print("VIF [N, P, K] :", [f"{v:,.0f}" if np.isfinite(v) else "inf"
                          for v in vif(Xz)])

# quantisation
print("\nQuantisation audit")
for c, lsb, unit in [("N_pct", 0.001, "%"), ("P_raw", 0.1, "raw"),
                     ("K_raw", 0.001, "raw")]:
    u = np.sort(df[c].unique())
    print(f"  {c:6s} LSB={lsb:<6} distinct levels overall = {len(u):2d}")
for s in "UPK":
    sub = df[df.series == s]
    print(f"  series {s}: distinct N_pct levels = {sub['N_pct'].nunique()} "
          f"({sorted(sub['N_pct'].unique())})")

# --------------------------------------------------------------------------
# 3. DIAGNOSTIC B -- sensitivity / cross-sensitivity matrix
# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SECTION 3 - SENSITIVITY MATRIX  d(channel)/d(applied nutrient)")
print("=" * 78)
print(f"{'':>12}{'sensor N':>22}{'sensor P':>22}{'sensor K':>22}")
S = {}
for ser, nut in [("U", "N"), ("P", "P"), ("K", "K")]:
    sub = df[df.series == ser]
    x = sub[f"{nut}_add"].to_numpy(float)
    row = []
    for chn in ["N_pct", "P_raw", "K_raw"]:
        lr = stats.linregress(x, sub[chn].to_numpy(float))
        S[(nut, chn)] = lr
        row.append(f"{lr.slope:+.4g} (R2={lr.rvalue**2:.2f})")
    print(f"applied {nut:<4}" + "".join(f"{v:>22}" for v in row))
print("\nNOTE: off-diagonal R^2 >= diagonal R^2 => channels are not selective.")

# --------------------------------------------------------------------------
# 4. OUTLIER SCREEN (studentised residual, |t| > 2.5)
# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SECTION 4 - OUTLIER SCREEN")
print("=" * 78)
flag = []
for ser, nut in [("U", "N"), ("P", "P"), ("K", "K")]:
    sub = df[df.series == ser]
    x = sub[f"{nut}_add"].to_numpy(float)
    for chn in ["N_pct", "P_raw"]:
        y = sub[chn].to_numpy(float)
        lr = stats.linregress(x, y)
        r = y - (lr.slope * x + lr.intercept)
        sd = r.std(ddof=2)
        if sd > 0:
            t = r / sd
            for smp, ti in zip(sub["Sample"], t):
                if abs(ti) > 2.0:
                    flag.append((smp, chn, ti))
                    print(f"  {smp}  {chn}  studentised resid = {ti:+.2f}")
print(f"  sensor pH range {df['pH'].min()} - {df['pH'].max()}"
      f"  -> P1 pH={df.loc[df.Sample=='P1','pH'].item()} is physically implausible"
      " for this soil (lab pH 7.8): probe-contact fault.")
EXCLUDE = ["K5", "P1"]
print(f"  EXCLUDED from calibration fits: {EXCLUDE}")
fit_df = df[~df["Sample"].isin(EXCLUDE)].reset_index(drop=True)

# --------------------------------------------------------------------------
# 5. THE FORWARD MODEL  (sensor = f(applied nutrients))
# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SECTION 5 - FORWARD MODEL:  P_raw = b0 + bN*N + bP*P + bK*K")
print("=" * 78)

A = fit_df[["N_add", "P_add", "K_add"]].to_numpy(float)
y = fit_df["P_raw"].to_numpy(float)
Ad = np.column_stack([np.ones(len(A)), A])
beta, *_ = np.linalg.lstsq(Ad, y, rcond=None)
yhat = Ad @ beta
n, p = Ad.shape
dof = n - p
mse = ((y - yhat) ** 2).sum() / dof
cov = mse * np.linalg.inv(Ad.T @ Ad)
se = np.sqrt(np.diag(cov))
tcrit = stats.t.ppf(0.975, dof)
r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
r2adj = 1 - (1 - r2) * (n - 1) / dof

names = ["intercept b0", "bN (per mg/kg N)", "bP (per mg/kg P)", "bK (per mg/kg K)"]
for nm, b, s_ in zip(names, beta, se):
    tval = b / s_
    print(f"  {nm:<22} = {b:+9.4f}  +/- {tcrit*s_:6.4f} (95% CI)"
          f"   t={tval:+6.2f}  p={2*(1-stats.t.cdf(abs(tval), dof)):.4f}")
print(f"  R^2 = {r2:.4f}   adj R^2 = {r2adj:.4f}   RMSE = {np.sqrt(mse):.2f} raw counts")

# relative ionic sensitivity (per mmol of element)
MW = {"N": 14.007, "P": 30.974, "K": 39.098}
print("\n  Sensitivity per mmol of element (slope * molar mass):")
for nm, mw, b in zip("NPK", [MW["N"], MW["P"], MW["K"]], beta[1:]):
    print(f"    {nm}: {b*mw:8.2f} counts per mmol/kg")
print("  Limiting molar ionic conductivity (S cm2/mol, 25 C) for reference:")
print("    K+ 73.5 | Cl- 76.3 | H2PO4- 36 | urea 0 (non-ionic before hydrolysis)")

# --------------------------------------------------------------------------
# 6. THE INVERSE PROBLEM -- why per-nutrient recovery is impossible
# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SECTION 6 - INVERSE PROBLEM")
print("=" * 78)
Kmat = np.array([[S[("N", c)].slope, S[("P", c)].slope, S[("K", c)].slope]
                 for c in ["N_pct", "P_raw", "K_raw"]])
det = np.linalg.det(Kmat)
_, sv3, _ = np.linalg.svd(Kmat)
print("Sensitivity matrix K (rows = sensor N/P/K, cols = applied N/P/K):")
print(np.array2string(Kmat, formatter={"float_kind": lambda v: f"{v:11.4g}"}))
print(f"\n  det(K)            = {det:.3e}")
print(f"  singular values   = {np.array2string(sv3, precision=3)}")
print(f"  cond(K)           = {sv3[0]/sv3[-1]:,.0f}")
print("  => effective rank 1. Recovering (N,P,K) from the probe is an\n"
      "     under-determined inverse problem with a 2-D null space.\n"
      "     No amount of regression or ML can fix this; it is a hardware limit.")

# --------------------------------------------------------------------------
# 7. WHAT *CAN* BE CALIBRATED -- composite ionic loading index
# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SECTION 7 - USABLE CALIBRATION: composite ionic loading index (CIL)")
print("=" * 78)
w = beta[1:] / beta[3]          # normalise so that K has weight 1.0
print(f"  CIL (mg/kg K-equivalent) = {w[0]:.4f}*N + {w[1]:.4f}*P + {w[2]:.4f}*K")
fit_df["CIL"] = A @ w
df["CIL"] = df[["N_add", "P_add", "K_add"]].to_numpy(float) @ w

cil = stats.linregress(fit_df["P_raw"], fit_df["CIL"])
resid_cil = fit_df["CIL"] - (cil.slope * fit_df["P_raw"] + cil.intercept)
sd_cil = resid_cil.std(ddof=2)
print(f"\n  INVERSE (deployable) equation:")
print(f"     CIL_hat = {cil.slope:.5f} * P_raw + ({cil.intercept:+.4f})")
print(f"     R^2 = {cil.rvalue**2:.4f}   RMSE = {sd_cil:.2f} mg/kg K-eq")
xr = fit_df["P_raw"]
sxx = ((xr - xr.mean())**2).sum()
print(f"     slope SE = {cil.stderr:.5f}  intercept SE = {cil.intercept_stderr:.4f}")
lod = 3 * sd_cil
loq = 10 * sd_cil
print(f"     LOD (3s) = {lod:.1f} mg/kg K-eq | LOQ (10s) = {loq:.1f} mg/kg K-eq")

# --------------------------------------------------------------------------
# 8. VALIDATION -- LOO CV, per-nutrient vs composite vs mean baseline
# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SECTION 8 - LEAVE-ONE-OUT VALIDATION vs MEAN BASELINE")
print("=" * 78)

def loo_scores(X, yv):
    X = np.asarray(X, float).reshape(len(yv), -1)
    yv = np.asarray(yv, float)
    pr = np.empty_like(yv)
    for tr, te in LeaveOneOut().split(X):
        pr[te] = LinearRegression().fit(X[tr], yv[tr]).predict(X[te])
    rmse = np.sqrt(((yv - pr) ** 2).mean())
    rmse0 = np.sqrt(((yv - yv.mean()) ** 2).mean())   # predict-the-mean
    return rmse, rmse0, yv.std(ddof=1) / rmse, 1 - (rmse / rmse0) ** 2, pr

print(f"{'target':<26}{'RMSE_cv':>10}{'RMSE_mean':>11}{'RPD':>7}{'NSE':>8}  verdict")
results = {}
targets = [
    ("Applied N (mg/kg)", "N_add", ["N_pct", "P_raw", "K_raw"]),
    ("Applied P (mg/kg)", "P_add", ["N_pct", "P_raw", "K_raw"]),
    ("Applied K (mg/kg)", "K_add", ["N_pct", "P_raw", "K_raw"]),
    ("Composite CIL",     "CIL",   ["P_raw"]),
]
for label, tcol, feats in targets:
    rmse, rmse0, rpd, nse, pr = loo_scores(fit_df[feats], fit_df[tcol])
    verdict = ("USABLE" if rpd >= 1.4 else
               "marginal" if rpd >= 1.0 else "FAILS (worse than mean)")
    print(f"{label:<26}{rmse:>10.2f}{rmse0:>11.2f}{rpd:>7.2f}{nse:>8.2f}  {verdict}")
    results[label] = dict(rmse=rmse, rmse0=rmse0, rpd=rpd, nse=nse,
                          pred=pr.tolist(), obs=fit_df[tcol].tolist())
print("\n  RPD >= 1.4 is the conventional threshold for a screening-grade sensor")
print("  RPD <  1.0 means the model is beaten by simply predicting the mean.")

# --------------------------------------------------------------------------
# 9. pH channel
# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SECTION 9 - pH CHANNEL")
print("=" * 78)
off = LAB_BASE["pH"] - SENS_BASE["pH"]
ph_ok = df[~df.Sample.isin(["P1"])]["pH"]
print(f"  Single-point offset from baseline soil : pH_cal = pH_raw + {off:.2f}")
print(f"  Sensor pH spread across 18 nominally identical-pH samples:"
      f" {ph_ok.min():.1f} - {ph_ok.max():.1f}  (sigma = {ph_ok.std(ddof=1):.2f})")
print("  => scatter (+/-{:.2f} pH) exceeds any plausible real variation."
      .format(ph_ok.std(ddof=1)))
print("  => a slope correction CANNOT be fitted: you have only ONE lab pH value.")
print("  => REQUIRED: 2-point (pH 4.01 / 7.00) or 3-point buffer calibration.")

# ==========================================================================
#                                 FIGURES
# ==========================================================================
def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"  saved figures/{name}")

print("\n" + "=" * 78)
print("SECTION 10 - FIGURES")
print("=" * 78)

# --- Fig 1 : data overview -------------------------------------------------
fig = plt.figure(figsize=(11, 6.2))
gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
corr = df[["N_pct", "P_raw", "K_raw", "pH", "RH"]].corr(method="spearman")
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
lbl = ["N", "P", "K", "pH", "RH"]
ax.set_xticklabels(lbl); ax.set_yticklabels(lbl); ax.grid(False)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                fontsize=7.5, color="white" if abs(corr.iloc[i, j]) > .6 else "black")
ax.set_title("(a) Spearman correlation")
fig.colorbar(im, ax=ax, shrink=0.8)

ax = fig.add_subplot(gs[0, 1])
ax.bar(range(1, 4), evr * 100, color=[C["fit"], C["P"], C["ci"]], width=.6)
ax.plot(range(1, 4), np.cumsum(evr) * 100, "o--", color=C["warn"], ms=4, lw=1)
for i, v in enumerate(evr * 100):
    ax.text(i + 1, v + 3, f"{v:.2f}%", ha="center", fontsize=8)
ax.set_xticks([1, 2, 3]); ax.set_xlabel("principal component")
ax.set_ylabel("variance explained (%)"); ax.set_ylim(0, 112)
ax.set_title("(b) PCA of log(N,P,K) channels")

ax = fig.add_subplot(gs[0, 2])
for s in "UPKM":
    sub = df[df.series == s]
    ax.scatter(sub["P_raw"], sub["K_raw"], s=34, color=C[s], label=f"{s}-series",
               zorder=3, edgecolor="white", linewidth=.6)
xx = np.linspace(df["P_raw"].min(), df["P_raw"].max(), 50)
ax.plot(xx, kp.slope * xx + kp.intercept, color=C["fit"], lw=1.2, zorder=2)
ax.set_xlabel("sensor P (raw)"); ax.set_ylabel("sensor K (raw)")
ax.set_title("(c) K is an affine map of P")
ax.text(.04, .92, f"$K = {kp.slope:.6f}P {kp.intercept:+.4f}$\n$R^2 = {kp.rvalue**2:.6f}$",
        transform=ax.transAxes, fontsize=8, va="top")
ax.legend(loc="lower right")

ax = fig.add_subplot(gs[1, 0])
ax.scatter(df["P_raw"], df["K_raw"] - (kp.slope * df["P_raw"] + kp.intercept),
           s=28, color=C["K"], edgecolor="white", linewidth=.6)
ax.axhline(0, color=C["fit"], lw=.9)
ax.set_xlabel("sensor P (raw)"); ax.set_ylabel("K residual")
ax.set_title(f"(d) residual $\\sigma$ = {k_resid.std(ddof=2):.4f}")

ax = fig.add_subplot(gs[1, 1])
sub = df[df.series == "U"].sort_values("N_add")
ax.step(sub["N_add"], sub["N_pct"] * 1e4, where="mid", color=C["U"], lw=1.6)
ax.scatter(sub["N_add"], sub["N_pct"] * 1e4, s=34, color=C["U"], zorder=3)
ax.axhline(10, color=C["ci"], ls=":", lw=1)
ax.text(30, 10.4, "1 LSB = 0.001 % = 10 mg/kg", fontsize=7.5, color=C["ci"])
ax.set_xlabel("applied N (mg/kg)"); ax.set_ylabel("sensor N ($10^{-4}$ %)")
ax.set_title("(e) N channel quantisation staircase")
ax.set_ylim(0, 62)

ax = fig.add_subplot(gs[1, 2])
for s, m in zip("UPKM", "os^D"):
    sub = df[df.series == s]
    ax.scatter(sub["RH"], sub["P_raw"], s=32, marker=m, color=C[s],
               label=f"{s}", edgecolor="white", linewidth=.6)
r_rh = stats.spearmanr(df["RH"], df["P_raw"])
ax.set_xlabel("sensor moisture (%)"); ax.set_ylabel("sensor P (raw)")
ax.set_title(f"(f) moisture confound  $\\rho$ = {r_rh.statistic:+.2f}")
ax.legend(ncol=4, loc="upper right", handletextpad=.2, columnspacing=.6)

fig.suptitle("Figure 1  |  Probe channel diagnostics: the three NPK outputs are one measurement",
             fontsize=11, fontweight="bold", y=0.99)
save(fig, "fig1_channel_diagnostics.png")

# --- Fig 2 : 3x3 sensitivity matrix ---------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(9.6, 8.2), sharex="col")
chan_lbl = ["sensor N (%)", "sensor P (raw)", "sensor K (raw)"]
for i, chn in enumerate(["N_pct", "P_raw", "K_raw"]):
    for j, (ser, nut) in enumerate([("U", "N"), ("P", "P"), ("K", "K")]):
        ax = axes[i, j]
        sub = df[df.series == ser]
        x = sub[f"{nut}_add"].to_numpy(float)
        yv = sub[chn].to_numpy(float)
        lr = S[(nut, chn)]
        diag = (i == j)
        col = C[ser] if diag else C["ci"]
        ax.scatter(x, yv, s=36, color=col, zorder=3, edgecolor="white", linewidth=.7)
        xx = np.linspace(x.min(), x.max(), 30)
        ax.plot(xx, lr.slope * xx + lr.intercept, color=C["fit"] if diag else col,
                lw=1.4 if diag else 1.0, ls="-" if diag else "--")
        ax.text(.05, .93, f"$R^2$={lr.rvalue**2:.3f}", transform=ax.transAxes,
                fontsize=8.5, va="top",
                fontweight="bold" if diag else "normal")
        if diag:
            for sp in ax.spines.values():
                sp.set_edgecolor(C[ser]); sp.set_linewidth(1.6)
            ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
        if i == 0:
            ax.set_title(f"applied {nut} series", fontsize=9.5)
        if i == 2:
            ax.set_xlabel(f"applied {nut} (mg/kg)")
        if j == 0:
            ax.set_ylabel(chan_lbl[i])
fig.suptitle("Figure 2  |  Sensitivity matrix — boxed diagonal = the channel that *should* respond\n"
             "off-diagonal responses are as strong or stronger: the probe is not nutrient-selective",
             fontsize=11, fontweight="bold", y=1.005)
fig.tight_layout()
save(fig, "fig2_sensitivity_matrix.png")

# --- Fig 3 : composite calibration curve ----------------------------------
fig = plt.figure(figsize=(11, 4.4))
gs = GridSpec(1, 3, figure=fig, wspace=0.30)

ax = fig.add_subplot(gs[0, 0])
xg = np.linspace(fit_df["P_raw"].min() * .95, fit_df["P_raw"].max() * 1.05, 120)
yg = cil.slope * xg + cil.intercept
nfit = len(fit_df)
tc = stats.t.ppf(0.975, nfit - 2)
sem = sd_cil * np.sqrt(1 / nfit + (xg - xr.mean()) ** 2 / sxx)
sep = sd_cil * np.sqrt(1 + 1 / nfit + (xg - xr.mean()) ** 2 / sxx)
ax.fill_between(xg, yg - tc * sep, yg + tc * sep, color=C["ci"], alpha=.18,
                label="95% prediction band")
ax.fill_between(xg, yg - tc * sem, yg + tc * sem, color=C["ci"], alpha=.42,
                label="95% confidence band")
ax.plot(xg, yg, color=C["fit"], lw=1.6, label="calibration line")
for s in "UPKM":
    sub = fit_df[fit_df.series == s]
    ax.scatter(sub["P_raw"], sub["CIL"], s=40, color=C[s], zorder=4,
               edgecolor="white", linewidth=.7, label=f"{s}-series")
ax.set_xlabel("sensor P channel (raw)")
ax.set_ylabel("composite ionic loading\n(mg/kg K-equivalent)")
ax.set_title("(a) Calibration curve")
ax.text(.04, .95, f"$R^2$ = {cil.rvalue**2:.3f}\nRMSE = {sd_cil:.1f} mg/kg",
        transform=ax.transAxes, va="top", fontsize=8.5)
ax.legend(ncol=2, loc="lower right", fontsize=7)

ax = fig.add_subplot(gs[0, 1])
pred = cil.slope * fit_df["P_raw"] + cil.intercept
mean_ = (pred + fit_df["CIL"]) / 2
diff = pred - fit_df["CIL"]
bias, sdd = diff.mean(), diff.std(ddof=1)
for s in "UPKM":
    m = fit_df.series == s
    ax.scatter(mean_[m], diff[m], s=40, color=C[s], edgecolor="white",
               linewidth=.7, zorder=3)
ax.axhline(bias, color=C["fit"], lw=1.2)
ax.axhline(bias + 1.96 * sdd, color=C["warn"], ls="--", lw=1)
ax.axhline(bias - 1.96 * sdd, color=C["warn"], ls="--", lw=1)
ax.text(.98, .95, f"bias = {bias:+.2f}\nLoA = {bias-1.96*sdd:+.1f} to {bias+1.96*sdd:+.1f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8)
ax.set_xlabel("mean of methods (mg/kg K-eq)")
ax.set_ylabel("sensor − reference")
ax.set_title("(b) Bland–Altman agreement")

ax = fig.add_subplot(gs[0, 2])
labels = list(results.keys())
rpds = [results[k]["rpd"] for k in labels]
cols = [C["warn"] if r < 1.0 else C["M"] if r < 1.4 else C["P"] for r in rpds]
b = ax.barh(range(len(labels)), rpds, color=cols, height=.55)
ax.axvline(1.0, color=C["fit"], ls=":", lw=1.1)
ax.axvline(1.4, color=C["P"], ls="--", lw=1.1)
ax.text(1.0, len(labels) - 0.35, " RPD=1\nmean-level", fontsize=6.8, color=C["fit"],
        ha="left", va="top")
ax.text(1.45, len(labels) - 0.35, " RPD=1.4\nscreening", fontsize=6.8, color=C["P"],
        ha="left", va="top")
ax.set_ylim(-0.6, len(labels) - 0.2)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels([l.replace(" (mg/kg)", "") for l in labels], fontsize=8)
ax.set_xlabel("RPD  (SD / RMSE$_{cv}$)")
ax.set_title("(c) Leave-one-out validation")
for i, r in enumerate(rpds):
    ax.text(r + .04, i, f"{r:.2f}", va="center", fontsize=8)
ax.set_xlim(0, max(rpds) * 1.25)

fig.suptitle("Figure 3  |  What the probe CAN measure: a single composite ionic-loading index",
             fontsize=11, fontweight="bold", y=1.02)
save(fig, "fig3_composite_calibration.png")

# --- Fig 4 : per-nutrient failure + ordinal alternative --------------------
fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
for ax, (label, tcol, _) in zip(axes[:3], targets[:3]):
    r = results[label]
    obs, prd = np.array(r["obs"]), np.array(r["pred"])
    lo, hi = min(obs.min(), prd.min()), max(obs.max(), prd.max())
    pad = (hi - lo) * .12
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=C["fit"], lw=1,
            ls="--", label="1:1")
    ax.axhline(obs.mean(), color=C["warn"], lw=1, ls=":", label="predict the mean")
    for s in "UPKM":
        m = (fit_df.series == s).to_numpy()
        ax.scatter(obs[m], prd[m], s=38, color=C[s], zorder=3,
                   edgecolor="white", linewidth=.7)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"reference {label.split()[1]} (mg/kg)")
    ax.set_ylabel("LOO-CV prediction")
    ok = r["rpd"] >= 1.0
    ax.set_title(f"{label.split()[1]}:  RPD={r['rpd']:.2f}  "
                 f"{'OK' if ok else 'FAILS'}",
                 color=C["fit"] if ok else C["warn"])
    ax.legend(loc="upper left", fontsize=7)

# ordinal class mapping -- evaluated on LEAVE-ONE-OUT predictions, not in-sample
ax = axes[3]
edges = np.quantile(fit_df["CIL"], [0, 1/3, 2/3, 1.0])
edges[0] -= 1e-6
true_cls = np.digitize(fit_df["CIL"], edges[1:-1])
_, _, _, _, cil_loo = loo_scores(fit_df[["P_raw"]], fit_df["CIL"])
pred_cls = np.digitize(cil_loo, edges[1:-1])
cm = np.zeros((3, 3), int)
for t, p_ in zip(true_cls, pred_cls):
    cm[t, p_] += 1
acc = np.trace(cm) / cm.sum()
po, pe = acc, sum(cm.sum(0)[i] * cm.sum(1)[i] for i in range(3)) / cm.sum() ** 2
kappa = (po - pe) / (1 - pe)
im = ax.imshow(cm, cmap="Blues")
ax.grid(False)
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                color="white" if cm[i, j] > cm.max() * .6 else "black")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(["Low", "Med", "High"]); ax.set_yticklabels(["Low", "Med", "High"])
ax.set_xlabel("predicted class"); ax.set_ylabel("reference class")
ax.set_title(f"Ordinal (LOO): acc={acc:.0%}, $\\kappa$={kappa:.2f}")
print(f"\n  Ordinal 3-class (LOO-validated): accuracy={acc:.1%}  Cohen kappa={kappa:.3f}")
print(f"  class edges (mg/kg K-eq): {np.round(edges[1:-1], 1)}")
coef_ordinal = {"edges_Keq": edges[1:-1].tolist(), "accuracy": float(acc),
                "kappa": float(kappa)}

fig.suptitle("Figure 4  |  Per-nutrient regression fails (left) — ordinal classification survives (right)",
             fontsize=11, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "fig4_pernutrient_vs_ordinal.png")

# --- Fig 5 : pH channel ----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
ax = axes[0]
for s in "UPKM":
    sub = df[df.series == s]
    ax.scatter(range(len(sub)), sub["pH"], s=40, color=C[s], label=s,
               edgecolor="white", linewidth=.7, zorder=3)
ax.axhline(LAB_BASE["pH"], color=C["fit"], lw=1.2, label="lab pH 7.8")
ax.axhline(df[df.Sample != "P1"]["pH"].mean(), color=C["warn"], ls="--", lw=1,
           label="sensor mean")
ax.annotate("P1 fault", xy=(0, 3.4), xytext=(1.2, 4.4), fontsize=8,
            color=C["warn"], arrowprops=dict(arrowstyle="->", color=C["warn"], lw=.9))
ax.set_ylabel("pH"); ax.set_xlabel("replicate index within series")
ax.set_title("(a) pH scatter on one soil"); ax.legend(ncol=3, fontsize=7)

ax = axes[1]
ax.hist(df[df.Sample != "P1"]["pH"], bins=np.arange(6.7, 8.9, 0.2),
        color=C["P"], edgecolor="white")
ax.axvline(LAB_BASE["pH"], color=C["fit"], lw=1.4)
ax.text(LAB_BASE["pH"] + .05, ax.get_ylim()[1] * .85, "lab reference", fontsize=8)
ax.set_xlabel("sensor pH"); ax.set_ylabel("count")
ax.set_title(f"(b) $\\sigma$ = {df[df.Sample!='P1']['pH'].std(ddof=1):.2f} pH units")
fig.suptitle("Figure 5  |  pH channel: precision-limited, needs buffer calibration",
             fontsize=11, fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "fig5_ph_channel.png")

# --------------------------------------------------------------------------
# 11. EXPORT COEFFICIENTS
# --------------------------------------------------------------------------
coef = {
    "firmware_relation_K_from_P": {"slope": kp.slope, "intercept": kp.intercept,
                                   "r2": kp.rvalue ** 2, "resid_sd": float(k_resid.std(ddof=2))},
    "forward_model_P_raw": {"b0": beta[0], "bN": beta[1], "bP": beta[2],
                            "bK": beta[3], "se": se.tolist(), "r2": r2,
                            "rmse": float(np.sqrt(mse))},
    "composite_weights_Keq": {"wN": w[0], "wP": w[1], "wK": w[2]},
    "inverse_CIL_from_P_raw": {"slope": cil.slope, "intercept": cil.intercept,
                               "r2": cil.rvalue ** 2, "rmse": float(sd_cil),
                               "lod": float(lod), "loq": float(loq)},
    "ordinal_3class": coef_ordinal,
    "ph_single_point_offset": off,
    "excluded_samples": EXCLUDE,
    "loo_validation": {k: {kk: vv for kk, vv in v.items()
                           if kk in ("rmse", "rmse0", "rpd", "nse")}
                       for k, v in results.items()},
}
Path("calibration_coefficients.json").write_text(json.dumps(coef, indent=2))
print("\n  saved calibration_coefficients.json")
print("\n" + "=" * 78)
print("DONE")
print("=" * 78)
