import io
import os
import math
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.pipeline import Pipeline

app = FastAPI(title="Machining Optimizer GPR API", version="1.0.0")

# Set FRONTEND_ORIGIN to your exact GitHub Pages origin after deployment.
# Example: https://yourusername.github.io
origins = [x.strip() for x in os.getenv("FRONTEND_ORIGIN", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

MATERIALS = {}
REQUIRED_COLUMNS = [
    "Cutting speed (m/min)", "Feed rate (mm/rev)", "Depth of cut (mm)",
    "Cutting angle (deg)", "Kc1.1 (N/mm^2)", "mc",
    "Cutting Power (kW)", "Cutting force (N)"
]

def kienzle_force(doc, feed_per_rev, cutting_angle_deg, kc11, mc):
    angle_rad = np.radians(cutting_angle_deg)
    if np.any(np.asarray(cutting_angle_deg) <= 0) or np.any(np.asarray(cutting_angle_deg) >= 180):
        raise ValueError("Cutting angle must be between 0 and 180 degrees.")
    sine = np.sin(angle_rad)
    if np.any(sine <= 0):
        raise ValueError("Invalid cutting angle: sin(angle) must be positive.")
    width_of_cut = doc / sine
    chip_thickness = feed_per_rev * sine
    return kc11 * width_of_cut * (chip_thickness ** (1 - mc))

def make_kernel():
    # Retained from the uploaded Python script.
    return (
        ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e2))
        * Matern(length_scale=np.ones(4), length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1.0))
    )

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
        df["Depth of cut (mm)"].to_numpy(),
        df["Feed rate (mm/rev)"].to_numpy(),
        df["Cutting angle (deg)"].to_numpy(),
        df["Kc1.1 (N/mm^2)"].to_numpy(),
        df["mc"].to_numpy()
    )
    X = df[[
        "Cutting speed (m/min)", "Feed rate (mm/rev)",
        "Depth of cut (mm)", "F_physics (N)"
    ]].to_numpy()
    y_force = df["Cutting force (N)"].to_numpy()
    y_power = df["Cutting Power (kW)"].to_numpy()

    X_train, X_test, F_train, F_test, P_train, P_test = train_test_split(
        X, y_force, y_power, test_size=0.2, random_state=42
    )

    # Keep the source script's 5-fold power cross-validation pipeline.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_power_model = Pipeline([
        ("scaler", StandardScaler()),
        ("gpr", GaussianProcessRegressor(
            kernel=make_kernel(), n_restarts_optimizer=5,
            normalize_y=True, random_state=42, alpha=1e-6
        ))
    ])
    scores = cross_validate(
        cv_power_model, X, y_power, cv=cv,
        scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
        n_jobs=1
    )

    # Fit scaling on training data only, as in the uploaded script.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    gpr_force = GaussianProcessRegressor(
        kernel=make_kernel(), n_restarts_optimizer=5,
        normalize_y=True, random_state=42, alpha=1e-6
    )
    gpr_force.fit(X_train_scaled, F_train)
    F_pred, F_std = gpr_force.predict(X_test_scaled, return_std=True)

    gpr_power = GaussianProcessRegressor(
        kernel=make_kernel(), n_restarts_optimizer=5,
        normalize_y=True, random_state=42, alpha=1e-6
    )
    gpr_power.fit(X_train_scaled, P_train)
    P_pred, P_std = gpr_power.predict(X_test_scaled, return_std=True)

    metrics = {
        "Force MAE (N)": float(mean_absolute_error(F_test, F_pred)),
        "Force RMSE (N)": float(np.sqrt(mean_squared_error(F_test, F_pred))),
        "Force R2": float(r2_score(F_test, F_pred)),
        "Power MAE (kW)": float(mean_absolute_error(P_test, P_pred)),
        "Power RMSE (kW)": float(np.sqrt(mean_squared_error(P_test, P_pred))),
        "Power R2": float(r2_score(P_test, P_pred)),
        "Power 5-fold CV MAE (kW)": float(-scores["test_mae"].mean()),
        "Power 5-fold CV R2": float(scores["test_r2"].mean())
    }
    return {
        "data": df, "scaler": scaler, "force_model": gpr_force,
        "power_model": gpr_power, "metrics": metrics,
        "row_count": int(len(df)),
        "force_test": F_test, "force_pred": F_pred, "force_std": F_std,
        "power_test": P_test, "power_pred": P_pred, "power_std": P_std
    }

class OptimizeRequest(BaseModel):
    material_name: str
    total_depth_mm: float = Field(gt=0)
    tool_diameter_mm: float = Field(gt=0)
    cut_length_per_pass_mm: Optional[float] = Field(default=None, gt=0)

@app.get("/")
def root():
    return {"service": "Machining Optimizer GPR API", "status": "ok", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "loaded_materials": len(MATERIALS)}

@app.get("/api/materials")
def list_materials():
    return {"materials": [
        {"name": name, "rows": model["row_count"], "metrics": model["metrics"]}
        for name, model in MATERIALS.items()
    ]}

@app.post("/api/materials")
async def upload_material(file: UploadFile = File(...), material_name: str = Form(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    name = material_name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Material name cannot be blank.")
    try:
        content = await file.read()
        # latin1 matches the encoding used in the original script and accepts common CSV exports.
        df = pd.read_csv(io.BytesIO(content), encoding="latin1")
        trained = train_one_material(name, df)
        MATERIALS[name] = trained
        return {"message": "GPR models trained", "name": name,
                "rows": trained["row_count"], "metrics": trained["metrics"]}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.delete("/api/materials")
def clear_materials():
    MATERIALS.clear()
    return {"message": "All in-memory material models cleared."}

@app.post("/api/optimize")
def optimize(req: OptimizeRequest):
    if req.material_name not in MATERIALS:
        raise HTTPException(status_code=404, detail="Material not found in backend memory. Upload its CSV and wait for GPR training first.")
    material_model = MATERIALS[req.material_name]
    material_df = material_model["data"]
    scaler = material_model["scaler"]
    gpr_power = material_model["power_model"]
    setting_columns = [
        "Cutting speed (m/min)", "Feed rate (mm/rev)", "Cutting angle (deg)",
        "Kc1.1 (N/mm^2)", "mc"
    ]
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
        pass_depths = [float(candidate_doc)] * (pass_count - 1) + [final_pass_doc]

        for cutting_speed, feed_per_rev, angle_deg, kc11, mc in setting_rows:
            spindle_rpm = 1000.0 * float(cutting_speed) / (np.pi * req.tool_diameter_mm)
            feed_mm_min = float(feed_per_rev) * spindle_rpm
            if feed_mm_min <= 0:
                continue
            pass_powers = []
            for actual_pass_doc in pass_depths:
                physics_force = kienzle_force(
                    actual_pass_doc, float(feed_per_rev), float(angle_deg), float(kc11), float(mc)
                )
                features = np.array([[
                    float(cutting_speed), float(feed_per_rev), actual_pass_doc, float(physics_force)
                ]])
                predicted_power, _ = gpr_power.predict(scaler.transform(features), return_std=True)
                pass_powers.append(max(float(predicted_power[0]), 0.0))
            average_power = float(np.mean(pass_powers))
            total_energy = None
            total_co2 = None
            if req.cut_length_per_pass_mm is not None:
                pass_time_hours = (req.cut_length_per_pass_mm / feed_mm_min) / 60.0
                total_energy = float(sum(power * pass_time_hours for power in pass_powers))
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
                "Total predicted energy (kWh)": total_energy,
                "Predicted CO2 emissions (kg)": total_co2,
                "_ranking_score": total_energy if total_energy is not None else average_power
            })
    results.sort(key=lambda row: row["_ranking_score"])
    clean_results = []
    for row in results[:5]:
        clean_results.append({k:v for k,v in row.items() if k != "_ranking_score"})
    return {
        "material": req.material_name,
        "energy_calculated": req.cut_length_per_pass_mm is not None,
        "validation_metrics": material_model["metrics"],
        "results": clean_results
    }
