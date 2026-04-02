import numpy as np
from pathlib import Path
import pandas as pd
from .excel_utils import read_motor_excel, infer_load_from_path, validate_feature_columns

def list_excel_files(root):
    root = Path(root)
    return sorted([p for p in root.rglob("*.xlsx") if p.is_file()])

def resolve_time_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None

def build_dataset_from_folder(root_folder, config):
    root_folder = Path(root_folder)
    files = list_excel_files(root_folder)
    if not files:
        raise FileNotFoundError(f"No se encontraron .xlsx en {root_folder}")

    frames = []
    time_candidates = config.get("time_column_candidates", ["Muestra", "Etiqueta_de_tiempo"])

    for file in files:
        df = read_motor_excel(
            file,
            sheet_name=config["sheet_name"],
            header_row_visual=config["header_row_visual"]
        )

        validate_feature_columns(df, config["feature_columns"])
        time_col = resolve_time_column(df, time_candidates)

        keep_cols = list(config["feature_columns"])

        if config.get("include_load_column", False):
            carga = infer_load_from_path(file, config["load_mapping_from_folder"])
            if carga is None:
                raise ValueError(f"No pude inferir la carga desde la ruta {file}")
            df[config["load_column_name"]] = float(carga)
            keep_cols.append(config["load_column_name"])

        out = df[keep_cols].copy()
        if time_col is not None:
            out["__time__"] = pd.to_numeric(df[time_col], errors="coerce")
        else:
            out["__time__"] = np.arange(len(df), dtype=float)
        out["__row_order__"] = np.arange(len(df), dtype=int)
        out["__source_file__"] = file.name
        out["__source_path__"] = str(file)
        out["__load_group__"] = file.parent.name
        frames.append(out)

    dataset = pd.concat(frames, ignore_index=True)
    return dataset

def split_by_file(dataset, train_ratio=0.8, seed=42):
    rng = np.random.RandomState(seed)
    file_groups = (
        dataset[["__source_file__", "__load_group__"]]
        .drop_duplicates()
        .groupby("__load_group__")["__source_file__"]
        .apply(list)
        .to_dict()
    )

    train_files = set()
    val_files = set()

    for _, files in sorted(file_groups.items()):
        files = list(files)
        rng.shuffle(files)
        n_train = max(1, int(len(files) * train_ratio))
        if n_train >= len(files):
            n_train = len(files) - 1
        train_files.update(files[:n_train])
        val_files.update(files[n_train:])

    train_df = dataset[dataset["__source_file__"].isin(train_files)].copy()
    val_df = dataset[dataset["__source_file__"].isin(val_files)].copy()

    return train_df, val_df, sorted(train_files), sorted(val_files)

def build_window_matrix(dataset, feature_cols, window_size=64, stride=16):
    windows = []
    window_meta = []

    for source_file, group in dataset.groupby("__source_file__", sort=False):
        ordered = group.sort_values(["__time__", "__row_order__"]).reset_index(drop=True)
        values = ordered[feature_cols].to_numpy(dtype=float)
        if len(values) < window_size:
            continue

        for start in range(0, len(values) - window_size + 1, stride):
            end = start + window_size
            windows.append(values[start:end].reshape(-1))
            window_meta.append(
                {
                    "source_file": source_file,
                    "start_idx": int(start),
                    "end_idx": int(end - 1),
                }
            )

    if not windows:
        raise ValueError(
            f"No se pudieron construir ventanas. window_size={window_size}, stride={stride}"
        )

    return np.asarray(windows, dtype=float), pd.DataFrame(window_meta)
