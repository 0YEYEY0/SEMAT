import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import torch

from src.dataset_builder import build_window_matrix, resolve_time_column
from src.excel_utils import read_motor_excel, validate_feature_columns, infer_load_from_path
from src.model import MLPAutoencoder
from src.training_utils import reconstruction_error


BASE = Path(__file__).resolve().parent


def main(filepath):
    config = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
    ckpt = torch.load(BASE / "models" / "autoencoder_motor.pt", map_location="cpu")
    scaler = joblib.load(BASE / "models" / "scaler.pkl")
    threshold_info = json.loads((BASE / "models" / "umbral.json").read_text(encoding="utf-8"))

    df = read_motor_excel(
        filepath,
        sheet_name=config["sheet_name"],
        header_row_visual=config["header_row_visual"]
    )

    validate_feature_columns(df, config["feature_columns"])

    feature_cols = list(config["feature_columns"])
    if config.get("include_load_column", False):
        carga = infer_load_from_path(Path(filepath), config["load_mapping_from_folder"])
        if carga is None:
            raise ValueError("No pude inferir la carga desde la ruta del archivo")
        df[config["load_column_name"]] = float(carga)
        feature_cols.append(config["load_column_name"])

    invalid_rows = df[feature_cols].isna().any(axis=1)
    dropped_rows = int(invalid_rows.sum())
    if dropped_rows:
        df = df.loc[~invalid_rows].copy()
        print(f"Filas descartadas por NaN en features: {dropped_rows}")

    mu = scaler["mu"]
    sigma = scaler["sigma"]

    df = df.copy()
    df.loc[:, feature_cols] = (df[feature_cols].to_numpy(dtype=float) - mu) / sigma

    time_candidates = config.get("time_column_candidates", ["Muestra", "Etiqueta_de_tiempo"])
    time_col = resolve_time_column(df, time_candidates)
    if time_col is not None:
        df["__time__"] = pd_to_numeric(df[time_col])
    else:
        df["__time__"] = np.arange(len(df), dtype=float)
    df["__row_order__"] = np.arange(len(df), dtype=int)
    df["__source_file__"] = Path(filepath).name
    df["__source_path__"] = str(Path(filepath))

    Xs, window_meta = build_window_matrix(
        df,
        feature_cols,
        window_size=int(ckpt["window_size"]),
        stride=int(ckpt.get("window_stride", ckpt["window_size"]))
    )

    model = MLPAutoencoder(input_dim=ckpt["input_dim"], hidden_dims=ckpt["hidden_dims"])
    model.load_state_dict(ckpt["state_dict"])

    err = reconstruction_error(model, Xs, device="cpu")
    threshold = threshold_info["threshold"]
    file_threshold = float(config.get("file_anomaly_fraction_threshold", 0.20))

    frac_anom = float(np.mean(err > threshold))
    verdict = "FALLA" if frac_anom > file_threshold else "SANO"

    print(f"Error medio: {float(np.mean(err)):.6f}")
    print(f"Fraccion anomala: {frac_anom:.4f}")
    print(f"Umbral fraccion archivo: {file_threshold:.4f}")
    print(f"Ventanas evaluadas: {len(window_meta)}")
    print(f"Veredicto: {verdict}")

    src_path = Path(filepath)
    dst_dir = BASE / "data" / ("detectado_falla" if verdict == "FALLA" else "detectado_sano")
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_dir / src_path.name)


def pd_to_numeric(series):
    return np.asarray(series, dtype=float)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Uso: python evaluate_motor.py data/por_evaluar/carga_100/archivo.xlsx")
    main(sys.argv[1])
