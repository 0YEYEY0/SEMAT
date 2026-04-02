import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_loader(x_np, batch_size=256, shuffle=True):
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    ds = TensorDataset(x_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def train_autoencoder(model, train_loader, val_loader, lr=1e-3, epochs=80, device="cpu"):
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    patience = 10
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, xb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                out = model(xb)
                loss = criterion(out, xb)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history

def reconstruction_error(model, x_np, device="cpu"):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_np, dtype=torch.float32).to(device)
        out = model(x_tensor)
        err = torch.mean((x_tensor - out) ** 2, dim=1)
    return err.cpu().numpy()

def save_threshold(path, threshold, percentile):
    path = Path(path)
    path.write_text(json.dumps({"threshold": float(threshold), "percentile": float(percentile)}, indent=2), encoding="utf-8")