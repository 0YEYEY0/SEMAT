import json
from pathlib import Path
import joblib
import numpy as np
import torch

from src.dataset_builder import build_dataset_from_folder, split_by_file, build_window_matrix
from src.model import MLPAutoencoder
from src.training_utils import set_seed, make_loader, train_autoencoder, reconstruction_error, save_threshold

BASE = Path(__file__).resolve().parent

def main():
    config = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
    set_seed(config["random_seed"])

    sano_root = BASE / "data" / "sanos"
    dataset = build_dataset_from_folder(sano_root, config)

    feature_cols = list(config["feature_columns"])
    if config.get("include_load_column", False):
        feature_cols.append(config["load_column_name"])

    invalid_rows = dataset[feature_cols].isna().any(axis=1)
    dropped_rows = int(invalid_rows.sum())
    if dropped_rows:
        dataset = dataset.loc[~invalid_rows].copy()
        print(f"Filas descartadas por NaN en features: {dropped_rows}")

    window_size = int(config.get("window_size", 64))
    window_stride = int(config.get("window_stride", 16))

    train_df, val_df, train_files, val_files = split_by_file(
        dataset,
        train_ratio=config["train_val_split_by_file"],
        seed=config["random_seed"]
    )

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    X_val = val_df[feature_cols].to_numpy(dtype=float)

    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0

    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df.loc[:, feature_cols] = (X_train - mu) / sigma
    val_df.loc[:, feature_cols] = (X_val - mu) / sigma

    X_train_s, train_windows = build_window_matrix(
        train_df,
        feature_cols,
        window_size=window_size,
        stride=window_stride
    )
    X_val_s, val_windows = build_window_matrix(
        val_df,
        feature_cols,
        window_size=window_size,
        stride=window_stride
    )

    train_loader = make_loader(X_train_s, batch_size=config["batch_size"], shuffle=True)
    val_loader = make_loader(X_val_s, batch_size=config["batch_size"], shuffle=False)

    input_dim = X_train_s.shape[1]
    model = MLPAutoencoder(input_dim=input_dim, hidden_dims=config["hidden_dims"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, history = train_autoencoder(
        model,
        train_loader,
        val_loader,
        lr=config["learning_rate"],
        epochs=config["epochs"],
        device=device
    )

    train_err = reconstruction_error(model, X_train_s, device=device)
    threshold = np.percentile(train_err, config["threshold_percentile"])

    models_dir = BASE / "models"
    models_dir.mkdir(exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dims": config["hidden_dims"],
            "feature_cols": feature_cols,
            "window_size": window_size,
            "window_stride": window_stride,
        },
        models_dir / "autoencoder_motor.pt"
    )

    joblib.dump({"mu": mu, "sigma": sigma}, models_dir / "scaler.pkl")
    save_threshold(models_dir / "umbral.json", threshold, config["threshold_percentile"])
    (models_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (models_dir / "split_info.json").write_text(
        json.dumps(
            {
                "train_files": train_files,
                "val_files": val_files,
                "train_windows": int(len(train_windows)),
                "val_windows": int(len(val_windows)),
                "window_size": window_size,
                "window_stride": window_stride,
                "dropped_rows": dropped_rows,
            },
            indent=2
        ),
        encoding="utf-8"
    )

    print(f"Archivos train: {len(train_files)}")
    print(f"Archivos val: {len(val_files)}")
    print(f"Ventanas train: {len(train_windows)}")
    print(f"Ventanas val: {len(val_windows)}")
    print(f"Umbral: {threshold:.6f}")

if __name__ == "__main__":
    main()
