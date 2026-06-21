#!/usr/bin/env python3
"""Train and evaluate torchvision image-classification baselines."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cervithink.constants import CERVICAL_LABELS
from cervithink.data import read_jsonl
from cervithink.metrics import classification_report
from cervithink.parsing import normalize_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CerviCoT/Ground-R1 JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arch", choices=("resnet50", "vit_b_16"), required=True)
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-model", action="store_true")
    return parser.parse_args()


def import_deps():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
    except Exception as exc:
        raise SystemExit(f"Missing vision baseline dependencies: {exc}")
    return torch, nn, DataLoader, Dataset, models, transforms


def resolve_image(path: str, image_root: str | None) -> str:
    image_path = Path(path)
    if image_root and not image_path.is_absolute():
        image_path = Path(image_root) / image_path
    return str(image_path)


def build_transform(transforms, weights_obj):
    if weights_obj is not None:
        return weights_obj.transforms()
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_model(arch: str, weights_name: str, num_classes: int, nn, models, transforms):
    weights_obj = None
    if arch == "resnet50":
        if weights_name == "imagenet":
            weights_obj = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights_obj)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "vit_b_16":
        if weights_name == "imagenet":
            weights_obj = models.ViT_B_16_Weights.IMAGENET1K_V1
        model = models.vit_b_16(weights=weights_obj)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return model, build_transform(transforms, weights_obj)


def split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("split", split) == split]


def main() -> None:
    args = parse_args()
    torch, nn, DataLoader, Dataset, models, transforms = import_deps()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = read_jsonl(args.input)
    train_rows = split_rows(rows, args.train_split)
    eval_rows = split_rows(rows, args.eval_split)
    if not train_rows:
        raise SystemExit(f"No rows found for train split {args.train_split!r}.")
    if not eval_rows:
        raise SystemExit(f"No rows found for eval split {args.eval_split!r}.")

    label_to_idx = {label: index for index, label in enumerate(CERVICAL_LABELS)}
    idx_to_label = {index: label for label, index in label_to_idx.items()}
    model, transform = build_model(args.arch, args.weights, len(CERVICAL_LABELS), nn, models, transforms)

    class CellDataset(Dataset):
        def __init__(self, records: list[dict[str, Any]]):
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int):
            row = self.records[index]
            image = Image.open(resolve_image(str(row["image"]), args.image_root)).convert("RGB")
            label = label_to_idx[normalize_label(str(row["label"]))]
            return transform(image), label, row

    def collate(batch):
        images, labels, batch_rows = zip(*batch)
        return torch.stack(list(images)), torch.tensor(labels, dtype=torch.long), list(batch_rows)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = DataLoader(
        CellDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    eval_loader = DataLoader(
        CellDataset(eval_rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for images, labels, _rows in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            seen += labels.size(0)
        print(f"epoch={epoch} loss={total_loss / max(seen, 1):.6f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / f"{args.arch}_predictions.jsonl"
    metrics_path = out_dir / f"{args.arch}_metrics.json"

    labels_out: list[str] = []
    predictions_out: list[str] = []
    model.eval()
    with predictions_path.open("w", encoding="utf-8") as handle, torch.no_grad():
        for images, labels, batch_rows in eval_loader:
            logits = model(images.to(device))
            pred_indices = logits.argmax(dim=1).cpu().tolist()
            for row, pred_index in zip(batch_rows, pred_indices):
                prediction = idx_to_label[int(pred_index)]
                labels_out.append(normalize_label(str(row["label"])))
                predictions_out.append(prediction)
                handle.write(
                    json.dumps(
                        {
                            "problem_id": row.get("problem_id"),
                            "image": row.get("image"),
                            "label": row.get("label"),
                            "prediction": prediction,
                            "model": args.arch,
                            "weights": args.weights,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    report = classification_report(labels_out, predictions_out)
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.save_model:
        torch.save(model.state_dict(), out_dir / f"{args.arch}.pt")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
