"""Fine-tune DistilBERT for three-class Amazon review sentiment.

Install the optional deep-learning dependencies, prepare the M3 data, and run
from the repository root:

    python -m pip install -r requirements-model.txt
    python src/prepare_m3_data.py
    python src/distilbert_model.py

The inner-training split selects the checkpoint and number of epochs. The
held-out test split is scored once by the final model. Prepared ``data/m3``
artifacts are used when present; otherwise the script recreates the same
M2/M3 splits from the shared CSV. See docs/M3_DISTILBERT.md for details.
"""

import argparse
import json
import random
import time
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

import pipeline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "m3"
DEFAULT_OUTPUT_DIR = ROOT / "models" / "distilbert"
DEFAULT_REPORT_DIR = ROOT / "results" / "m3_distilbert"
DEFAULT_FIGURE_DIR = ROOT / "figures"
DEFAULT_MODEL = "distilbert-base-uncased"
LABELS = pipeline.LABELS
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
SPLIT_NAMES = ("outer_train", "test", "inner_train", "validation")
COL = {"negative": "#2a78d6", "neutral": "#eb6834", "positive": "#1baf7a"}


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def load_splits(data_dir=DEFAULT_DATA_DIR):
    """Load the locked M3 splits, falling back to the shared M2 sample.

    Prepared artifacts keep the exact row order used by the M3 handoff. The
    fallback repeats the same seed-47 split so the deep model remains directly
    comparable with the classical models.
    """
    data_dir = Path(data_dir)
    reviews_path = data_dir / "reviews.jsonl"
    if reviews_path.is_file():
        missing = [name for name in SPLIT_NAMES
                   if not (data_dir / f"{name}_labels.csv").is_file()]
        if missing:
            raise ValueError(
                f"Prepared data is incomplete in {data_dir}; missing label files "
                f"for: {', '.join(missing)}. Rerun python src/prepare_m3_data.py."
            )
        reviews = pd.read_json(reviews_path, lines=True).set_index("Id", drop=False)
        if reviews.index.duplicated().any():
            raise ValueError("Prepared reviews contain duplicate Id values.")
        splits = {}
        for name in SPLIT_NAMES:
            ordering = pd.read_csv(data_dir / f"{name}_labels.csv")
            frame = reviews.loc[ordering["Id"].tolist()].reset_index(drop=True)
            if frame["Id"].tolist() != ordering["Id"].tolist():
                raise ValueError(f"{name} row order does not match its label file.")
            expected = frame["label"].map(LABEL_TO_ID).to_numpy(dtype=int)
            if "label_id" not in frame or not np.array_equal(
                expected, ordering["label_id"].to_numpy(dtype=int)
            ):
                raise ValueError(f"{name} labels disagree with the prepared label IDs.")
            splits[name] = frame
        source = f"prepared M3 artifacts ({data_dir})"
    else:
        if not pipeline.DATA.is_file():
            raise SystemExit(
                f"Training data not found. Run python src/prepare_m3_data.py, or "
                f"create {pipeline.DATA.name} with python sample.py path/to/Reviews.csv."
            )
        data, _ = pipeline.load()
        outer_train, test = pipeline.split_data(data)
        inner_train, validation = train_test_split(
            outer_train, test_size=0.20, stratify=outer_train["label"],
            random_state=pipeline.SEED,
        )
        splits = dict(outer_train=outer_train, test=test,
                      inner_train=inner_train, validation=validation)
        source = f"shared training CSV ({pipeline.DATA})"
    verify_splits(splits)
    return splits, source


def verify_splits(splits):
    """Reject overlapping or malformed train/validation/test frames."""
    missing_splits = set(SPLIT_NAMES).difference(splits)
    if missing_splits:
        raise ValueError(f"Missing splits: {', '.join(sorted(missing_splits))}")
    required = {"Id", "Summary", "Text", "label"}
    for name in SPLIT_NAMES:
        frame = splits[name]
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {', '.join(sorted(missing))}")
        if frame.empty:
            raise ValueError(f"{name} is empty.")
        if frame["Id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate review IDs.")
        if not frame["label"].isin(LABELS).all():
            raise ValueError(f"{name} contains labels outside {LABELS}.")
        if frame["Text"].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} contains empty review bodies.")

    ids = {name: set(splits[name]["Id"].tolist()) for name in SPLIT_NAMES}
    if ids["outer_train"] & ids["test"]:
        raise ValueError("Train/test review IDs overlap.")
    if ids["inner_train"] & ids["validation"]:
        raise ValueError("Inner-training/validation review IDs overlap.")
    if ids["inner_train"] | ids["validation"] != ids["outer_train"]:
        raise ValueError("Inner-training and validation rows must partition outer training.")
    if "clean_body" in splits["outer_train"] and "clean_body" in splits["test"]:
        if set(splits["outer_train"]["clean_body"]) & set(splits["test"]["clean_body"]):
            raise ValueError("Train/test cleaned review bodies overlap.")


def text_pairs(frame):
    """Return cleaned summary/body lists for DistilBERT's sentence-pair input.

    The M2 cleaner removes HTML/URL artifacts and normalizes whitespace. Its
    lowercasing matches the default uncased checkpoint; punctuation is kept.
    """
    if not isinstance(frame, pd.DataFrame) or "Text" not in frame:
        raise ValueError("Expected a DataFrame with a Text column.")
    summary = (frame["Summary"].fillna("").astype(str).map(pipeline.clean)
               if "Summary" in frame
               else pd.Series("", index=frame.index, dtype="object"))
    body = frame["Text"].fillna("").astype(str).map(pipeline.clean)
    return summary.tolist(), body.tolist()


def split_summary(frame):
    counts = frame["label"].value_counts().reindex(LABELS, fill_value=0)
    return {"rows": int(len(frame)), "labels": {label: int(counts[label]) for label in LABELS}}


def compute_class_weights(labels, enabled=True):
    """Return balanced cross-entropy weights, or None when disabled."""
    if not enabled:
        return None
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        raise ValueError("Cannot compute class weights for an empty label array.")
    counts = np.bincount(labels, minlength=len(LABELS))[:len(LABELS)]
    weights = np.zeros(len(LABELS), dtype=np.float64)
    present = counts > 0
    weights[present] = labels.size / (len(LABELS) * counts[present])
    return weights.tolist()


def classification_metrics(y_true, y_pred, loss=None):
    """Summarize predictions with the same label order as the M2 results."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    class_ids = list(range(len(LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_ids, zero_division=0,
    )
    result = {
        "n": int(len(y_true)),
        "labels": list(LABELS),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=class_ids,
                                  average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=class_ids,
                                     average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_ids).tolist(),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
    }
    if loss is not None:
        result["loss"] = float(loss)
    return result


def prediction_frame(frame, probabilities, predicted_ids):
    """Create a CSV-ready prediction table with one probability per class."""
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(frame), len(LABELS)):
        raise ValueError("Probability matrix shape does not match the review frame.")
    output = frame[["Id", "label"]].reset_index(drop=True).copy()
    output["predicted"] = [LABELS[int(index)] for index in predicted_ids]
    for index, label in enumerate(LABELS):
        output[f"prob_{label}"] = probabilities[:, index]
    output["correct"] = output["label"].eq(output["predicted"])
    return output


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_training_figure(history, best_epoch, path):
    """Save loss and validation macro-F1 curves for the M3 report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].plot(epochs, [row["train_loss"] for row in history],
                 color=COL["negative"], marker="o", label="training")
    axes[0].plot(epochs, [row["validation_loss"] for row in history],
                 color=COL["neutral"], marker="o", label="validation")
    axes[0].set_title("DistilBERT loss", loc="left")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, [row["validation_macro_f1"] for row in history],
                 color=COL["positive"], marker="o")
    axes[1].axvline(best_epoch, color="#52514e", linestyle="--", linewidth=0.8)
    axes[1].set_title("Validation macro-F1", loc="left")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#e4e3df", linewidth=0.6)
        axis.set_axisbelow(True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def import_training_dependencies():
    """Import optional heavy dependencies only when training is requested."""
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer,
                                  get_linear_schedule_with_warmup)
    except ImportError as error:
        raise SystemExit(
            "DistilBERT dependencies are not installed. Run "
            "python -m pip install -r requirements-model.txt"
        ) from error
    return {
        "torch": torch,
        "DataLoader": DataLoader,
        "TensorDataset": TensorDataset,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
    }


def set_seed(torch, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(torch, requested=None):
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise SystemExit("MPS was requested but is not available.")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(deps, tokenizer, frame, max_length, batch_size, shuffle,
                num_workers, pin_memory):
    summaries, bodies = text_pairs(frame)
    encoding = tokenizer(
        summaries, bodies, truncation=True, max_length=max_length,
        padding="max_length", return_tensors="pt",
    )
    labels = deps["torch"].tensor(
        frame["label"].map(LABEL_TO_ID).to_numpy(dtype=int),
        dtype=deps["torch"].long,
    )
    dataset = deps["TensorDataset"](
        encoding["input_ids"], encoding["attention_mask"], labels,
    )
    return deps["DataLoader"](
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )


def build_model(deps, model_name, device):
    # Fine-tuning replaces the pretrained classifier head with our three labels.
    # Transformers 5 raises on that expected size mismatch unless allowed.
    return deps["AutoModelForSequenceClassification"].from_pretrained(
        model_name, num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id=LABEL_TO_ID, ignore_mismatched_sizes=True,
    ).to(device)


def make_loss_fn(deps, class_weights, device):
    torch = deps["torch"]
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)
    return torch.nn.CrossEntropyLoss(weight=weight_tensor)


def make_training_components(deps, model, train_loader, epochs, args,
                             class_weights, device):
    torch = deps["torch"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = deps["get_linear_schedule_with_warmup"](
        optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    loss_fn = make_loss_fn(deps, class_weights, device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    return optimizer, scheduler, loss_fn, scaler, use_amp


def train_one_epoch(deps, model, loader, optimizer, scheduler, loss_fn,
                    device, scaler, use_amp):
    torch = deps["torch"]
    model.train()
    total_loss, total_rows = 0.0, 0
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        autocast = (torch.amp.autocast("cuda", enabled=True)
                    if use_amp else nullcontext())
        with autocast:
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * labels.size(0)
        total_rows += labels.size(0)
    return total_loss / max(1, total_rows)


def evaluate_model(deps, model, loader, loss_fn, device):
    torch = deps["torch"]
    model.eval()
    total_loss, total_rows = 0.0, 0
    probabilities, targets = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_rows += labels.size(0)
            probabilities.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
            targets.append(labels.cpu().numpy())
    probabilities = np.concatenate(probabilities)
    targets = np.concatenate(targets)
    return targets, probabilities.argmax(axis=1), probabilities, total_loss / max(1, total_rows)


def fit_with_validation(deps, model, train_loader, validation_loader, args,
                        class_weights, device, checkpoint_path):
    torch = deps["torch"]
    optimizer, scheduler, loss_fn, scaler, use_amp = make_training_components(
        deps, model, train_loader, args.epochs, args, class_weights, device,
    )
    history = []
    best_macro_f1, best_epoch, stale_epochs = -1.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss = train_one_epoch(
            deps, model, train_loader, optimizer, scheduler, loss_fn,
            device, scaler, use_amp,
        )
        y_true, y_pred, _, validation_loss = evaluate_model(
            deps, model, validation_loader, loss_fn, device,
        )
        metrics = classification_metrics(y_true, y_pred, validation_loss)
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "validation_loss": float(validation_loss),
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "epoch_seconds": float(time.perf_counter() - started),
        }
        history.append(row)
        print(
            f"epoch {epoch:2d}/{args.epochs} train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} val_acc={metrics['accuracy']:.4f} "
            f"val_macroF1={metrics['macro_f1']:.4f} ({row['epoch_seconds']:.1f}s)",
            flush=True,
        )
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
            if args.patience and stale_epochs >= args.patience:
                print(f"Early stopping after {stale_epochs} epoch(s) without improvement.")
                break
    if best_epoch:
        model.load_state_dict(torch.load(
            checkpoint_path, map_location=device, weights_only=True,
        ))
    checkpoint_path.unlink(missing_ok=True)
    return model, history, best_epoch, best_macro_f1


def fit_final(deps, model, train_loader, epochs, args, class_weights, device):
    optimizer, scheduler, loss_fn, scaler, use_amp = make_training_components(
        deps, model, train_loader, epochs, args, class_weights, device,
    )
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            deps, model, train_loader, optimizer, scheduler, loss_fn,
            device, scaler, use_amp,
        )
        print(f"final epoch {epoch:2d}/{epochs} train_loss={train_loss:.4f}", flush=True)
    return model


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Prepared M3 directory (default: data/m3).")
    parser.add_argument("--model-name", default=DEFAULT_MODEL,
                        help="Hugging Face model name or local checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Where to save the final model and tokenizer.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR,
                        help="Where to save metrics, predictions, and metadata.")
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR,
                        help="Where to save the loss-curve figure.")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Maximum tokenized sequence length (default: 256).")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Maximum epochs for the validation run (default: 4).")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=1,
                        help="Early-stopping patience; 0 disables early stopping.")
    parser.add_argument("--seed", type=int, default=pipeline.SEED)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None,
                        help="Force a device; by default CUDA, MPS, then CPU is used.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Use unweighted cross-entropy instead of balanced weights.")
    parser.add_argument("--no-final-fit", action="store_true",
                        help="Score the inner-trained checkpoint on test instead of "
                             "refitting on all outer training reviews.")
    return parser


def validate_args(parser, args):
    if args.max_length < 8:
        parser.error("--max-length must be at least 8.")
    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        parser.error("Batch sizes must be positive.")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative.")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("--warmup-ratio must be in [0, 1).")
    if args.patience < 0:
        parser.error("--patience cannot be negative.")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")


def run(args):
    deps = import_training_dependencies()
    torch = deps["torch"]
    set_seed(torch, args.seed)
    device = select_device(torch, args.device)
    started = time.perf_counter()
    print(f"Using device: {device}", flush=True)

    splits, data_source = load_splits(args.data_dir)
    print(f"Data source: {data_source}", flush=True)
    tokenizer = deps["AutoTokenizer"].from_pretrained(args.model_name, use_fast=True)
    if args.max_length > tokenizer.model_max_length:
        raise SystemExit(
            f"--max-length {args.max_length} exceeds the tokenizer limit "
            f"{tokenizer.model_max_length} for {args.model_name}."
        )
    print(
        "Splits: " + ", ".join(
            f"{name}={len(frame):,}" for name, frame in splits.items()
        ), flush=True,
    )

    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        deps, tokenizer, splits["inner_train"], args.max_length,
        args.batch_size, True, args.num_workers, pin_memory,
    )
    validation_loader = make_loader(
        deps, tokenizer, splits["validation"], args.max_length,
        args.eval_batch_size, False, args.num_workers, pin_memory,
    )
    class_weights = compute_class_weights(
        splits["inner_train"]["label"].map(LABEL_TO_ID).to_numpy(),
        enabled=not args.no_class_weights,
    )
    if class_weights is not None:
        print("Class weights: " + ", ".join(
            f"{label}={weight:.3f}" for label, weight in zip(LABELS, class_weights)
        ), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(deps, args.model_name, device)
    model, history, best_epoch, best_validation_f1 = fit_with_validation(
        deps, model, train_loader, validation_loader, args, class_weights,
        device, args.output_dir / "best_validation.pt",
    )

    y_true, y_pred, probabilities, validation_loss = evaluate_model(
        deps, model, validation_loader,
        make_loss_fn(deps, class_weights, device), device,
    )
    validation_metrics = classification_metrics(y_true, y_pred, validation_loss)
    validation_metrics.update(
        split="validation", model=args.model_name, selected_epoch=best_epoch,
        class_weights=class_weights,
    )
    save_json(args.report_dir / "validation_metrics.json", validation_metrics)
    prediction_frame(splits["validation"], probabilities, y_pred).to_csv(
        args.report_dir / "validation_predictions.csv", index=False,
    )
    pd.DataFrame(history).to_csv(args.report_dir / "history.csv", index=False)
    save_training_figure(
        history, best_epoch, args.figure_dir / "distilbert_loss.png",
    )
    print(
        f"Best validation epoch {best_epoch}: macroF1={best_validation_f1:.4f}",
        flush=True,
    )

    final_split = "inner_train"
    final_class_weights = class_weights
    if args.no_final_fit:
        final_model = model
    else:
        final_split = "outer_train"
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        outer_weights = compute_class_weights(
            splits["outer_train"]["label"].map(LABEL_TO_ID).to_numpy(),
            enabled=not args.no_class_weights,
        )
        final_class_weights = outer_weights
        outer_loader = make_loader(
            deps, tokenizer, splits["outer_train"], args.max_length,
            args.batch_size, True, args.num_workers, pin_memory,
        )
        final_model = build_model(deps, args.model_name, device)
        final_model = fit_final(
            deps, final_model, outer_loader, best_epoch, args,
            outer_weights, device,
        )

    test_loader = make_loader(
        deps, tokenizer, splits["test"], args.max_length,
        args.eval_batch_size, False, args.num_workers, pin_memory,
    )
    test_weights = compute_class_weights(
        splits[final_split]["label"].map(LABEL_TO_ID).to_numpy(),
        enabled=not args.no_class_weights,
    )
    test_loss_fn = make_loss_fn(deps, test_weights, device)
    y_true, y_pred, probabilities, test_loss = evaluate_model(
        deps, final_model, test_loader, test_loss_fn, device,
    )
    test_metrics = classification_metrics(y_true, y_pred, test_loss)
    test_metrics.update(
        split="test", model=args.model_name, final_fit_split=final_split,
    )
    save_json(args.report_dir / "test_metrics.json", test_metrics)
    prediction_frame(splits["test"], probabilities, y_pred).to_csv(
        args.report_dir / "test_predictions.csv", index=False,
    )

    final_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "model_name": args.model_name,
        "data_source": data_source,
        "device": str(device),
        "seed": args.seed,
        "max_length": args.max_length,
        "class_weights": class_weights,
        "final_class_weights": final_class_weights,
        "settings": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio, "patience": args.patience,
        },
        "splits": {name: split_summary(frame) for name, frame in splits.items()},
        "selected_epoch": best_epoch,
        "best_validation_macro_f1": best_validation_f1,
        "final_fit_split": final_split,
        "epochs_run": len(history),
        "wall_seconds": float(time.perf_counter() - started),
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "pandas": package_version("pandas"),
            "numpy": package_version("numpy"),
            "scikit-learn": package_version("scikit-learn"),
        },
        "artifacts": {
            "model": str(args.output_dir),
            "report": str(args.report_dir),
            "figure": str(args.figure_dir / "distilbert_loss.png"),
        },
        "test_scored_once": True,
    }
    save_json(args.report_dir / "run_metadata.json", metadata)
    save_json(args.output_dir / "run_metadata.json", metadata)
    print(
        f"Test accuracy={test_metrics['accuracy']:.4f} "
        f"macroF1={test_metrics['macro_f1']:.4f}",
        flush=True,
    )
    print(f"Saved model to {args.output_dir}", flush=True)
    print(f"Saved metrics to {args.report_dir}", flush=True)


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    run(args)


if __name__ == "__main__":
    main()
