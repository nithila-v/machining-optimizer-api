import os
import asyncio
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.stats import norm
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

app = FastAPI(title="Machining Optimizer GPR API", version="1.1.0")

# Set FRONTEND_ORIGIN to your exact GitHub Pages origin after deployment.
origins = [x.strip() for x in os.getenv("FRONTEND_ORIGIN", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

MATERIALS = {}
MODEL_LOADING = False
MODEL_LOAD_ERROR = None
DATASET_DIR = Path(__file__).resolve().parent / "datasets"

MATERIAL_DATASETS = {
    "TiAl6V4": "final dataset.csv",
    "C45E": "dataset_two.csv",
}

REQUIRED_COLUMNS = [
    "Cutting speed (m/min)", "Feed rate (mm/rev)", "Depth of cut (mm)",
    "Cutting angle (deg)", "Kc1.1 (N/mm^2)", "mc",
    "Cutting Power (kW)", "Cutting force (N)"
]

# Columns that define one machining setting. Rows sharing all of these are
# treated as repeats of the same experiment when no explicit run-ID column exists.
SETTING_COLUMNS = [
    "Cutting speed (m/min)", "Feed rate (mm/rev)", "Depth of cut (mm)",
    "Cutting angle (deg)", "Kc1.1 (N/mm^2)", "mc"
]

# If the CSV has a run/experiment identifier, it is used for grouped splitting.
# Override with env GROUP_COLUMN (exact header). Names are matched case-insensitively.
GROUP_COLUMN_CANDIDATES = ["run id", "run", "experiment", "experiment id", "trial", "test no", "test id"]

# "residual": GP models (measured - Kienzle) and adds Kienzle back  -> physics as prior mean (primary)
# "feature" : Kienzle force is a 4th GP input (the original design)  -> ablation
# "none"    : plain 3-input GP, no physics                           -> ablation
PRIMARY_MODE = "residual"
ALL_MODES = ["residual", "feature", "none"]
RUN_ABLATION = os.getenv("RUN_ABLATION", "0") == "0"
COVERAGE_LEVELS = [0.50, 0.80, 0.90, 0.95]


def kienzle_force(doc, feed_per_rev, cutting_angle_deg, kc11, mc):
    if np.any(np.asarray(cutting_angle_deg) <= 0) or np.any(np.asarray(cutting_angle_deg) >= 180):
        raise ValueError("Cutting angle must be between 0 and 180 degrees.")
    angle_rad = np.radians(cutting_angle_deg)
    sine = np.sin(angle_rad)
    if np.any(sine <= 0):
        raise ValueError("Invalid cutting angle: sin(angle) must be positive.")
    width_of_cut = doc / sine
    chip_thickness = feed_per_rev * sine
    return kc11 * width_of_cut * (chip_thickness ** (1 - mc))


def kienzle_power_kw(force_n, speed_m_min):
    """P [kW] = Fc [N] * vc [m/min] / 60000. Check this matches how the dataset's power column was computed."""
    return np.asarray(force_n) * np.asarray(speed_m_min) / 60000.0


def make_kernel(n_features: int):
    return (
        ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e2))
        * Matern(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1.0))
    )


class PhysicsGPR:
    """GP regressor with three physics-handling modes (see PRIMARY_MODE comments)."""

    def __init__(self, mode: str, target: str):
        assert mode in ALL_MODES and target in ("force", "power")
        self.mode, self.target = mode, target

    def _features(self, speed, feed, doc, f_phys):
        cols = [speed, feed, doc]
        if self.mode == "feature":
            cols.append(f_phys)
        return np.column_stack(cols)

    def _prior(self, speed, f_phys):
        if self.mode != "residual":
            return 0.0
        return np.asarray(f_phys) if self.target == "force" else kienzle_power_kw(f_phys, speed)

    def fit(self, speed, feed, doc, f_phys, y):
        X = self._features(speed, feed, doc, f_phys)
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.gpr = GaussianProcessRegressor(
            kernel=make_kernel(Xs.shape[1]), n_restarts_optimizer=0,
            normalize_y=True, random_state=42, alpha=1e-6
        )
        self.gpr.fit(Xs, np.asarray(y) - self._prior(speed, f_phys))
        return self

    def predict(self, speed, feed, doc, f_phys):
        Xs = self.scaler.transform(self._features(speed, feed, doc, f_phys))
        mu, std = self.gpr.predict(Xs, return_std=True)
        return mu + self._prior(speed, f_phys), std


def _cols(df):
    return (df["Cutting speed (m/min)"].to_numpy(), df["Feed rate (mm/rev)"].to_numpy(),
            df["Depth of cut (mm)"].to_numpy(), df["F_physics (N)"].to_numpy())


def fit_predict(mode, target, df, train_idx, test_idx):
    ycol = "Cutting force (N)" if target == "force" else "Cutting Power (kW)"
    tr, te = df.iloc[train_idx], df.iloc[test_idx]
    model = PhysicsGPR(mode, target).fit(*_cols(tr), tr[ycol].to_numpy())
    mu, std = model.predict(*_cols(te))
    return model, te[ycol].to_numpy(), mu, std


def coverage_report(y, mu, std):
    """Empirical coverage of Gaussian predictive intervals vs. nominal level."""
    out = {}
    for level in COVERAGE_LEVELS:
        z = norm.ppf(0.5 + level / 2.0)
        out[f"{int(level * 100)}%"] = float(np.mean(np.abs(y - mu) <= z * std))
    z95 = norm.ppf(0.975)
    out["mean_95%_interval_width"] = float(np.mean(2 * z95 * std))
    out["n_points"] = int(len(y))
    return out


def point_metrics(y, mu):
    return {
        "MAE": float(mean_absolute_error(y, mu)),
        "RMSE": float(np.sqrt(mean_squared_error(y, mu))),
        "R2": float(r2_score(y, mu)),
    }


def find_groups(df):
    """Return (group labels, description). Prefer an explicit run/experiment column."""
    explicit = os.getenv("GROUP_COLUMN")
    lookup = {c.lower(): c for c in df.columns}
    col = explicit if explicit in df.columns else next(
        (lookup[c] for c in GROUP_COLUMN_CANDIDATES if c in lookup), None)
    if col is not None:
        return pd.factorize(df[col].astype(str))[0], f"column '{col}'"
    return df.groupby(SETTING_COLUMNS, sort=False).ngroup().to_numpy(), "identical machining settings (no run-ID column found)"


def train_one_material(material_name: str, df: pd.DataFrame):
    df = df.copy()
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing CSV columns: " + ", ".join(missing))
    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    if len(df) < 10:
        raise ValueError(f"Only {len(df)} complete numeric rows remain; at least 10 are required.")

    df["F_physics (N)"] = kienzle_force(
        df["Depth of cut (mm)"].to_numpy(), df["Feed rate (mm/rev)"].to_numpy(),
        df["Cutting angle (deg)"].to_numpy(), df["Kc1.1 (N/mm^2)"].to_numpy(),
        df["mc"].to_numpy()
    )

    # ---- Leakage-safe splitting: whole experiments/settings go to train OR test ----
    groups, group_desc = find_groups(df)
    n_groups = len(np.unique(groups))
    if n_groups < 6:
        raise ValueError(f"Only {n_groups} distinct experiments/settings; need at least 6 for grouped CV.")
    split_info = {
        "grouping": group_desc, "n_rows": int(len(df)), "n_groups": int(n_groups),
        "grouping_changes_split": bool(n_groups < len(df)),
    }

    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                               .split(df, groups=groups))

    # ---- Primary models (physics as prior mean) on the held-out groups ----
    force_model, F_test, F_pred, F_std = fit_predict(PRIMARY_MODE, "force", df, train_idx, test_idx)
    power_model, P_test, P_pred, P_std = fit_predict(PRIMARY_MODE, "power", df, train_idx, test_idx)

    # ---- Grouped 5-fold CV, out-of-fold predictions for both targets ----
    n_splits = min(5, n_groups)
    oof = {t: {"y": [], "mu": [], "std": []} for t in ("force", "power")}
    for tr, te in GroupKFold(n_splits=n_splits).split(df, groups=groups):
        for t in ("force", "power"):
            _, y, mu, sd = fit_predict(PRIMARY_MODE, t, df, tr, te)
            oof[t]["y"].append(y); oof[t]["mu"].append(mu); oof[t]["std"].append(sd)
    oof = {t: {k: np.concatenate(v) for k, v in d.items()} for t, d in oof.items()}

    pm_f, pm_p = point_metrics(F_test, F_pred), point_metrics(P_test, P_pred)
    cv_p = point_metrics(oof["power"]["y"], oof["power"]["mu"])
    cv_f = point_metrics(oof["force"]["y"], oof["force"]["mu"])
    metrics = {
        "Force MAE (N)": pm_f["MAE"], "Force RMSE (N)": pm_f["RMSE"], "Force R2": pm_f["R2"],
        "Power MAE (kW)": pm_p["MAE"], "Power RMSE (kW)": pm_p["RMSE"], "Power R2": pm_p["R2"],
        f"Power grouped {n_splits}-fold CV MAE (kW)": cv_p["MAE"],
        f"Power grouped {n_splits}-fold CV R2": cv_p["R2"],
        f"Force grouped {n_splits}-fold CV MAE (N)": cv_f["MAE"],
        f"Force grouped {n_splits}-fold CV R2": cv_f["R2"],
    }

    # ---- Interval calibration: does nominal X% actually contain X% of points? ----
    calibration = {
        "note": "Empirical coverage of Gaussian predictive intervals (mean +/- z*sigma) vs nominal level. "
                "Held-out = single group split; out_of_fold = grouped CV pooled over all rows.",
        "force": {"held_out": coverage_report(F_test, F_pred, F_std),
                  "out_of_fold": coverage_report(**{"y": oof["force"]["y"], "mu": oof["force"]["mu"], "std": oof["force"]["std"]})},
        "power": {"held_out": coverage_report(P_test, P_pred, P_std),
                  "out_of_fold": coverage_report(**{"y": oof["power"]["y"], "mu": oof["power"]["mu"], "std": oof["power"]["std"]})},
    }

    # ---- Ablation: residual-on-Kienzle vs Kienzle-as-feature vs no physics (same split) ----
    ablation = None
    if RUN_ABLATION:
        ablation = {}
        for mode in ALL_MODES:
            row = {}
            for t, unit in (("force", "N"), ("power", "kW")):
                if mode == PRIMARY_MODE:
                    y, mu, sd = (F_test, F_pred, F_std) if t == "force" else (P_test, P_pred, P_std)
                else:
                    _, y, mu, sd = fit_predict(mode, t, df, train_idx, test_idx)
                pm = point_metrics(y, mu)
                row[f"{t} MAE ({unit})"] = pm["MAE"]
                row[f"{t} R2"] = pm["R2"]
                row[f"{t} 95% coverage"] = coverage_report(y, mu, sd)["95%"]
            ablation[mode] = row

    return {
        "data": df, "force_model": force_model, "power_model": power_model,
        "metrics": metrics, "calibration": calibration, "ablation": ablation,
        "split_info": split_info, "row_count": int(len(df)),
    }


class OptimizeRequest(BaseModel):
    material_name: str
    total_depth_mm: float = Field(gt=0)
    workpiece_diameter_mm: float = Field(gt=0)
    cut_length_per_pass_mm: Optional[float] = Field(default=None, gt=0)


@app.get("/")
def root():
    return {"service": "Machining Optimizer GPR API", "status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    if MODEL_LOADING:
        return {"status": "loading_models", "loaded_materials": len(MATERIALS)}
    if MODEL_LOAD_ERROR:
        return {"status": "model_load_error", "loaded_materials": len(MATERIALS), "error": MODEL_LOAD_ERROR}
    return {"status": "ok", "loaded_materials": len(MATERIALS)}


@app.get("/api/materials")
def list_materials():
    return {"materials": [
        {"name": name, "rows": m["row_count"], "metrics": m["metrics"],
         "calibration": m["calibration"], "ablation": m["ablation"], "split_info": m["split_info"]}
        for name, m in MATERIALS.items()
    ]}


def load_bundled_materials():
    MATERIALS.clear()
    missing_files = []
    for material_name, filename in MATERIAL_DATASETS.items():
        csv_path = DATASET_DIR / filename
        if not csv_path.is_file():
            missing_files.append(str(csv_path))
            continue
        try:
            df = pd.read_csv(csv_path, encoding="latin1")
            MATERIALS[material_name] = train_one_material(material_name, df)
        except Exception as exc:
            raise RuntimeError(f"Could not load dataset for {material_name} ({filename}): {exc}") from exc
    if missing_files:
        for missing_path in missing_files:
            print(f"WARNING: material dataset not found: {missing_path}")
        print("Add the configured CSV files to backend/datasets/ and redeploy to enable those materials.")


def _load_materials_in_background():
    global MODEL_LOADING, MODEL_LOAD_ERROR
    try:
        load_bundled_materials()
        if not MATERIALS:
            MODEL_LOAD_ERROR = (
                "No material models loaded. Check that configured CSV files exist "
                "under backend/datasets/ and contain the required columns."
            )
        else:
            MODEL_LOAD_ERROR = None
    except Exception as exc:
        MODEL_LOAD_ERROR = str(exc)
        print(f"ERROR while loading material models: {exc}", flush=True)
    finally:
        MODEL_LOADING = False
        print(f"Material initialization finished. Loaded {len(MATERIALS)} model(s).", flush=True)


@app.on_event("startup")
async def startup_load_materials():
    global MODEL_LOADING, MODEL_LOAD_ERROR
    MODEL_LOADING = True
    MODEL_LOAD_ERROR = None
    asyncio.create_task(asyncio.to_thread(_load_materials_in_background))


@app.post("/api/optimize")
def optimize(req: OptimizeRequest):
    if MODEL_LOADING:
        raise HTTPException(status_code=503,
                            detail="Material models are still training after startup. Please wait briefly and try again.")
    if MODEL_LOAD_ERROR and not MATERIALS:
        raise HTTPException(status_code=503, detail=f"Material models failed to load: {MODEL_LOAD_ERROR}")
    if req.material_name not in MATERIALS:
        raise HTTPException(status_code=404,
                            detail="Material model is not loaded. Check the bundled datasets and backend logs.")
    material_model = MATERIALS[req.material_name]
    material_df = material_model["data"]
    gpr_power: PhysicsGPR = material_model["power_model"]
    setting_columns = ["Cutting speed (m/min)", "Feed rate (mm/rev)", "Cutting angle (deg)", "Kc1.1 (N/mm^2)", "mc"]
    setting_rows = material_df[setting_columns].drop_duplicates().to_numpy()
    doc_candidates = np.sort(material_df["Depth of cut (mm)"].unique())
    results = []
    for candidate_doc in doc_candidates:
        if candidate_doc <= 0:
            continue
        pass_count = int(np.ceil(req.total_depth_mm / candidate_doc))
        final_pass_doc = float(round(req.total_depth_mm - candidate_doc * (pass_count - 1), 8))
        if final_pass_doc <= 0:
            pass_count -= 1
            final_pass_doc = float(candidate_doc)
        pass_depths = np.array([float(candidate_doc)] * (pass_count - 1) + [final_pass_doc])
        n = len(pass_depths)

        for cutting_speed, feed_per_rev, angle_deg, kc11, mc in setting_rows:
            spindle_rpm = 1000.0 * float(cutting_speed) / (np.pi * req.workpiece_diameter_mm)
            feed_mm_min = float(feed_per_rev) * spindle_rpm
            if feed_mm_min <= 0:
                continue
            physics_force = kienzle_force(pass_depths, float(feed_per_rev), float(angle_deg), float(kc11), float(mc))
            mu, sd = gpr_power.predict(
                np.full(n, float(cutting_speed)), np.full(n, float(feed_per_rev)), pass_depths, physics_force
            )
            pass_powers = np.maximum(mu, 0.0)
            average_power = float(np.mean(pass_powers))
            mean_sigma = float(np.mean(sd))
            total_energy = None
            total_co2 = None
            if req.cut_length_per_pass_mm is not None:
                pass_time_hours = (req.cut_length_per_pass_mm / feed_mm_min) / 60.0
                total_energy = float(np.sum(pass_powers) * pass_time_hours)
                total_co2 = total_energy * 0.71
            results.append({
                "Material": req.material_name,
                "DOC option (mm)": float(candidate_doc),
                "Pass count": pass_count,
                "Final pass DOC (mm)": final_pass_doc,
                "Spindle speed (RPM)": float(spindle_rpm),
                "Feed rate (mm/min)": float(feed_mm_min),
                "Cutting angle (deg)": float(angle_deg),
                "Average predicted power (kW)": average_power,
                "Predictive sigma (kW, 1 sd)": mean_sigma,
                # Heuristic only: sigma relative to the prediction. NOT a statistical confidence level.
                "Relative uncertainty (%)": (mean_sigma / average_power * 100.0) if average_power > 0 else None,
                "Total predicted energy (kWh)": total_energy,
                "Predicted CO2 emissions (kg)": total_co2,
                "_ranking_score": total_energy if total_energy is not None else average_power
            })
    results.sort(key=lambda row: row["_ranking_score"])
    clean_results = [{k: v for k, v in row.items() if k != "_ranking_score"} for row in results[:5]]
    return {
        "material": req.material_name,
        "energy_calculated": req.cut_length_per_pass_mm is not None,
        "validation_metrics": material_model["metrics"],
        "calibration": material_model["calibration"],
        "split_info": material_model["split_info"],
        "results": clean_results
    }
