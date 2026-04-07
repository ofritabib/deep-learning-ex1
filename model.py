import os
import argparse
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from collections import Counter
from tqdm.auto import tqdm
import plotly.graph_objects as go

from data_prep import (
    build_dataset,
    build_protein_loader,
)


# ── Model ─────────────────────────────────────────────────────────────────────

class SimpleClassifier(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int, hidden_layer_dim: list[int], dropout: float = 0.3, act_fn=nn.ReLU):
        super().__init__()
        dims = [num_inputs] + list(hidden_layer_dim) + [num_outputs]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if act_fn is not None:
                    layers.append(act_fn())
                layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Loss ──────────────────────────────────────────────────────────────────────

def build_loss_module(
    dataset,
    neg_multiplier: float = 3.0,
) -> nn.CrossEntropyLoss:
    """
    CrossEntropyLoss with inverse-frequency class weights.
    The negative class weight is additionally boosted by neg_multiplier
    to make the model more conservative (fewer false positives).
    """
    raw_labels  = [y for _, y in dataset.samples]
    counts      = Counter(raw_labels)
    num_classes = dataset.num_classes
    total       = sum(counts.values())
    neg_idx     = num_classes - 1

    weights = torch.tensor(
        [total / (num_classes * counts[i]) for i in range(num_classes)],
        dtype=torch.float32,
    )
    weights[neg_idx] *= neg_multiplier
    print("Class weights (after neg boost):", weights)
    return nn.CrossEntropyLoss(weight=weights)


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataset,
    test_dataset,
    loss_module: nn.Module,
    num_epochs: int = 50,
    batch_size: int = 256,
    device: torch.device | str = "cpu",
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 10,
    scheduler_min_lr: float = 1e-5,
) -> tuple[list[float], list[float]]:
    """
    Train with CrossEntropyLoss + ReduceLROnPlateau scheduler.
    Returns (train_losses, test_losses) per epoch.
    """
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    model.to(device)
    loss_module = loss_module.to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor,
        patience=scheduler_patience, min_lr=scheduler_min_lr,
    )

    train_losses, test_losses = [], []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_train_loss = []
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            loss = loss_module(model(X), Y)
            epoch_train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_losses.append(float(np.mean(epoch_train_loss)))

        model.eval()
        epoch_test_loss = []
        with torch.no_grad():
            for X, Y in test_loader:
                X, Y = X.to(device), Y.to(device)
                epoch_test_loss.append(loss_module(model(X), Y).item())
        mean_test = float(np.mean(epoch_test_loss))
        test_losses.append(mean_test)
        scheduler.step(mean_test)

    print(f"Training complete. Final LR: {scheduler.get_last_lr()[0]:.2e}")
    return train_losses, test_losses


def report_losses(train_losses: list[float], test_losses: list[float], save_path: str) -> None:
    epochs = list(range(1, len(train_losses) + 1))
    fig = go.Figure()
    fig.add_scatter(x=epochs, y=train_losses, name="Train loss", mode="lines+markers")
    fig.add_scatter(x=epochs, y=test_losses,  name="Test loss",  mode="lines+markers")
    fig.update_layout(
        title="7-class CE loss over epochs",
        xaxis_title="Epoch",
        yaxis_title="Loss",
    )
    fig.write_html(f"{save_path}/losses.html")
    print(f"Loss plot saved to {save_path}/losses.html")

    losses_data = {"train_losses": train_losses, "test_losses": test_losses}
    with open(f"{save_path}/losses.json", "w") as f:
        json.dump(losses_data, f, indent=2)
    print(f"Losses saved to {save_path}/losses.json")


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    test_dataset,
    dataset,
    device: torch.device | str = "cpu",
    batch_size: int = 512,
    save_path: str = "metrics.json",
) -> dict:
    """
    Run evaluation on test_dataset.
    Returns a dict with overall accuracy, binary precision/recall/F1, and per-class accuracy.
    """
    model.eval()
    neg_class_idx = dataset.num_classes - 1

    all_preds, all_true = [], []
    with torch.no_grad():
        for X, Y in data.DataLoader(test_dataset, batch_size=batch_size):
            X, Y = X.to(device), Y.to(device)
            all_preds.append(model(X).argmax(dim=1).cpu())
            all_true.append(Y.argmax(dim=1).cpu())

    preds  = torch.cat(all_preds)
    y_true = torch.cat(all_true)

    overall_acc = (preds == y_true).float().mean().item()

    y_pos_pred = preds  != neg_class_idx
    y_pos_true = y_true != neg_class_idx
    tp = ( y_pos_pred &  y_pos_true).sum().item()
    fp = ( y_pos_pred & ~y_pos_true).sum().item()
    fn = (~y_pos_pred &  y_pos_true).sum().item()
    tn = (~y_pos_pred & ~y_pos_true).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    per_class = {}
    for i, cls in enumerate(dataset.classes):
        mask = y_true == i
        if mask.sum() == 0:
            continue
        per_class[cls] = {
            "accuracy": (preds[mask] == i).float().mean().item(),
            "n": mask.sum().item(),
        }

    results = dict(
        overall_acc=overall_acc,
        precision=precision, recall=recall, f1=f1,
        tp=tp, fp=fp, fn=fn, tn=tn,
        per_class=per_class,
    )

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Metrics saved to {save_path}")

    print(f"Overall 7-class accuracy: {overall_acc:.4f}")
    print(f"\nPos vs Neg (binary):")
    print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"\nPer-class accuracy:")
    for cls, v in per_class.items():
        print(f"  {cls:20s}  {v['accuracy']:.4f}  (n={v['n']})")

    return results


# ── protein inference & plotting ─────────────────────────────────────────────────

def evaluate_protein(
    model: nn.Module,
    dataset,
    protein_path: str,
    device: torch.device,
    batch_size: int = 512,
    save_dir: str = ".",
) -> None:
    """Run sliding-window inference on a protein FASTA and plot binding predictions."""
    protein_name = protein_path.split("/")[-1].replace(".txt", "")
    loader, peptides, positions, protein_len = build_protein_loader(protein_path, batch_size=batch_size)
    neg_class_idx = dataset.num_classes - 1

    os.makedirs(f"{save_dir}/{protein_name}", exist_ok=True)

    model.eval()
    all_logits = []
    with torch.no_grad():
        for X, _ in loader:
            all_logits.append(model(X.to(device)).cpu())

    logits     = torch.cat(all_logits)
    pred_class = logits.argmax(dim=1)
    is_pos     = pred_class != neg_class_idx
    probs      = F.softmax(logits, dim=1)
    pos_conf   = (1 - probs[:, neg_class_idx]).numpy()

    n_pos = is_pos.sum().item()
    print(f"Predicted positive: {n_pos}/{len(peptides)} ({100*n_pos/len(peptides):.1f}%)")

    allele_names = [
        c.replace("_pos.txt", "").replace("negs.txt", "neg")
        for c in dataset.classes
    ]
    pos_mask = is_pos.numpy().astype(bool)
    colors   = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "lightgray"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[p for p, m in zip(positions, pos_mask) if not m],
        y=[s for s, m in zip(pos_conf,  pos_mask) if not m],
        mode="markers", name="Predicted negative",
        marker=dict(color="lightgray", size=5, opacity=0.5),
        hovertemplate="pos %{x}<br>p(pos)=%{y:.3f}<extra>neg</extra>",
    ))
    for i, allele in enumerate(allele_names[:-1]):
        idx_mask = [m and pred_class[j].item() == i for j, m in enumerate(pos_mask)]
        fig.add_trace(go.Scatter(
            x=[p for p, m in zip(positions, idx_mask) if m],
            y=[s for s, m in zip(pos_conf,  idx_mask) if m],
            mode="markers", name=allele,
            marker=dict(color=colors[i], size=7),
            hovertemplate=f"pos %{{x}}  {allele}<br>p(pos)=%{{y:.3f}}<extra></extra>",
        ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="threshold", annotation_position="right")
    fig.update_layout(
        title=f"{protein_name} — predicted allele binding per 9-mer position",
        xaxis_title="Start position in protein",
        yaxis_title="P(positive) = 1 − P(neg class)",
        yaxis=dict(range=[0, 1]), height=450,
    )
    binding_path = f"{save_dir}/{protein_name}/binding.html"
    fig.write_html(binding_path)
    print(f"Binding plot saved to {binding_path}")

    # Coverage
    covered = set()
    window = 9
    for pos, is_p in zip(positions, pos_mask):
        if is_p:
            covered.update(range(pos, pos + window))
    coverage_pct = 100 * len(covered) / protein_len
    print(f"\nCovered positions: {len(covered)}/{protein_len} ({coverage_pct:.1f}%)")

    coverage_track = [1 if (i + 1) in covered else 0 for i in range(protein_len)]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=list(range(1, protein_len + 1)), y=coverage_track,
        marker_color=["crimson" if c else "lightgray" for c in coverage_track],
        hovertemplate="pos %{x}<extra></extra>",
    ))
    fig2.update_layout(
        title=f"{protein_name} coverage — {coverage_pct:.1f}% covered by a predicted binder",
        xaxis_title="Position in protein",
        yaxis=dict(visible=False), bargap=0, height=250, showlegend=False,
    )
    coverage_path = f"{save_dir}/{protein_name}/coverage.html"
    fig2.write_html(coverage_path)
    print(f"Coverage plot saved to {coverage_path}")

    top5 = sorted(
        zip(pos_conf.tolist(), positions, peptides, pred_class.tolist()), reverse=True
    )[:5]
    print("\nTop-5 predicted binders:")
    for rank, (p, pos, seq, cls) in enumerate(top5, 1):
        print(f"  {rank}. pos {pos:4d}  {seq}  {allele_names[cls]}  p={p:.3f}")

    protein_metrics = dict(
        protein=protein_name,
        n_peptides=len(peptides),
        n_positive=n_pos,
        pct_positive=round(100 * n_pos / len(peptides), 2),
        protein_len=protein_len,
        covered_positions=len(covered),
        coverage_pct=round(coverage_pct, 2),
        top5=[
            dict(rank=rank, position=pos, peptide=seq, allele=allele_names[cls], confidence=round(p, 4))
            for rank, (p, pos, seq, cls) in enumerate(top5, 1)
        ],
        per_position=[
            dict(position=pos, peptide=pep, predicted_class=allele_names[cls], confidence=round(float(conf), 4))
            for pos, pep, cls, conf in zip(positions, peptides, pred_class.tolist(), pos_conf.tolist())
        ],
    )
    metrics_path = f"{save_dir}/{protein_name}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(protein_metrics, f, indent=2)
    print(f"Protein metrics saved to {metrics_path}")


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Loaded config from {args.config}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ds_cfg    = cfg["dataset"]
    model_cfg = cfg["model"]
    loss_cfg  = cfg["loss"]
    train_cfg = cfg["training"]
    eval_cfg  = cfg["evaluation"]
    out_cfg   = cfg["output"]

    os.makedirs(out_cfg["output_dir"], exist_ok=True)

    dataset, aug_train_dataset, test_dataset = build_dataset(
        data_dir=ds_cfg["data_dir"],
        clean=ds_cfg["clean"],
        test_size=ds_cfg["test_size"],
        p_augment=ds_cfg["p_augment"],
        aug_temperature=ds_cfg["aug_temperature"],
    )
    print(f"Train: {len(aug_train_dataset)}  |  Test: {len(test_dataset)}")

    loss_module = build_loss_module(dataset, neg_multiplier=loss_cfg["neg_multiplier"])

    act_fn = getattr(nn, model_cfg["act_fn"])
    model  = SimpleClassifier(
        num_inputs=model_cfg["num_inputs"],
        num_outputs=dataset.num_classes,
        hidden_layer_dim=model_cfg["hidden_layer_dim"],
        dropout=model_cfg["dropout"],
        act_fn=act_fn,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    train_losses, test_losses = train(
        model, optimizer,
        aug_train_dataset, test_dataset,
        loss_module,
        num_epochs=train_cfg["num_epochs"],
        batch_size=train_cfg["batch_size"],
        device=device,
        scheduler_factor=train_cfg["scheduler_factor"],
        scheduler_patience=train_cfg["scheduler_patience"],
        scheduler_min_lr=train_cfg["scheduler_min_lr"],
    )
    report_losses(train_losses, test_losses, save_path=out_cfg["output_dir"])

    model_path = out_cfg["output_dir"] + "/model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    evaluate(
        model, test_dataset, dataset,
        device=device,
        batch_size=eval_cfg["batch_size"],
        save_path=out_cfg["output_dir"] + "/metrics.json",
    )

    for protein_path in eval_cfg.get("protein_paths", []):
        evaluate_protein(
            model, dataset,
            protein_path=protein_path,
            device=device,
            batch_size=eval_cfg["batch_size"],
            save_dir=out_cfg["output_dir"] + "/protein_evaluation",
        )


if __name__ == "__main__":
    main()
