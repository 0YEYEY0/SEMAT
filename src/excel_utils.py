import re
from pathlib import Path

import numpy as np
import pandas as pd


HEADER_HINTS = {
    "muestra",
    "rol_eje",
    "rol_abanico",
    "carcaza_abanico",
    "carcaza_centro",
    "carcaza_eje",
    "nucleo",
    "t_ambiente",
    "rtd_a",
    "rtd_b",
    "rtd_c",
    "velocidad",
    "torque",
    "nucleo_2",
    "potencia_in",
    "potencia_out",
    "eficiencia",
    "voltaje_a",
    "voltage_b",
    "voltage_c",
    "corriente_a",
    "corriente_b",
    "corriente_c",
    "fpotencia",
    "tdh_a",
    "tdh_b",
    "tdh_c",
    "desbalance",
}


def normalize_col(name):
    name = str(name).strip().replace("\n", " ").replace("\r", " ")
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^0-9A-Za-z_]", "", name)
    return name


def to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    x = str(x).strip().replace(",", ".").replace("âˆ’", "-")
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", x)
    return float(m.group(0)) if m else np.nan


def infer_load_from_path(path, mapping):
    path = Path(path)
    path_parts = [p.lower() for p in path.parts]
    for folder_name, value in mapping.items():
        if folder_name.lower() in path_parts:
            return float(value)
    return None


def _select_sheet(xls, sheet_name):
    target = sheet_name.strip().lower()
    for sheet in xls.sheet_names:
        if sheet.strip().lower() == target:
            return sheet
    if len(xls.sheet_names) == 1:
        return xls.sheet_names[0]
    raise ValueError(
        f"No encontre una hoja equivalente a '{sheet_name}' en {Path(xls.io).name}. "
        f"Hojas disponibles: {xls.sheet_names}"
    )


def _is_header_like(values):
    normalized = {
        normalize_col(value).lower()
        for value in values
        if isinstance(value, str) and value.strip()
    }
    return len(normalized & HEADER_HINTS) >= 5


def _parse_inline_csv_sheet(raw_df):
    for row_idx, row in raw_df.iterrows():
        values = [value for value in row.tolist() if not pd.isna(value)]
        for value in values:
            text = str(value).strip()
            if "," not in text:
                continue

            header = [normalize_col(part) for part in text.split(",")]
            if len({part.lower() for part in header} & HEADER_HINTS) < 5:
                continue

            parsed_rows = []
            for next_idx in range(row_idx + 1, len(raw_df)):
                next_values = [v for v in raw_df.iloc[next_idx].tolist() if not pd.isna(v)]
                if not next_values:
                    continue

                line = str(next_values[0]).strip()
                if "," not in line:
                    continue

                parts = [part.strip() for part in line.split(",")]
                if len(parts) != len(header):
                    continue
                parsed_rows.append(parts)

            if parsed_rows:
                return pd.DataFrame(parsed_rows, columns=header)
    return None


def _parse_tabular_sheet(raw_df, fallback_header_row):
    header_row_idx = None
    for row_idx, row in raw_df.iterrows():
        values = [value for value in row.tolist() if not pd.isna(value)]
        if _is_header_like(values):
            header_row_idx = row_idx
            break

    if header_row_idx is None:
        header_row_idx = max(0, fallback_header_row - 1)

    header_values = raw_df.iloc[header_row_idx].tolist()
    header = [normalize_col(value) for value in header_values]
    data = raw_df.iloc[header_row_idx + 1 :].copy()
    data.columns = header
    return data


def read_motor_excel(filepath, sheet_name="Sheet1", header_row_visual=2):
    filepath = Path(filepath)
    xls = pd.ExcelFile(filepath)
    real_sheet = _select_sheet(xls, sheet_name)
    raw_df = pd.read_excel(filepath, sheet_name=real_sheet, header=None)

    df = _parse_inline_csv_sheet(raw_df)
    if df is None:
        df = _parse_tabular_sheet(raw_df, header_row_visual)

    df.columns = [normalize_col(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    df = df.dropna(axis=1, how="all")
    for col in df.columns:
        df[col] = df[col].map(to_float)
    df = df.dropna(axis=0, how="all")
    return df


def validate_feature_columns(df, feature_columns):
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")
