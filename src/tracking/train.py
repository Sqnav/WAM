from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from tracking.data import UAVTrackingDataset
from tracking.losses import tracking_loss
from tracking.model import UAVTracker


def distributed_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return float(tensor.item())


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch: int, best_iou: float, args) -> None:
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    payload = {
        "model": state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_iou": best_iou,
        "args": vars(args),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_epoch(model, loader, optimizer, scaler, device, epoch: int, training: bool, print_interval: int) -> Dict[str, float]:
    model.train(training)
    totals = {key: 0.0 for key in ("total", "location", "l1", "giou", "iou")}
    seen = 0
    started = time.time()
    context = torch.enable_grad if training else torch.no_grad
    for step, batch in enumerate(loader, 1):
        template = batch["template"].to(device, non_blocking=True)
        search = batch["search"].to(device, non_blocking=True)
        target = batch["box"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with context(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(template, search)
            core = model.module if hasattr(model, "module") else model
            losses = tracking_loss(core, outputs, target)
        if training:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()
        batch_size = int(target.shape[0])
        seen += batch_size
        for key in totals:
            totals[key] += float(losses[key].detach()) * batch_size
        if print_interval and step % print_interval == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            speed = seen / max(time.time() - started, 1e-6)
            print(
                f"{'train' if training else 'val'} epoch={epoch:03d} step={step:05d}/{len(loader):05d} "
                f"loss={totals['total']/seen:.4f} iou={totals['iou']/seen:.4f} samples/s={speed:.2f}",
                flush=True,
            )
    global_seen = distributed_sum(seen, device)
    return {key: distributed_sum(value, device) / max(global_seen, 1.0) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single-model UAV visual tracker")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", default="deit_tiny_patch16_224")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--pretrained-path", type=Path)
    parser.add_argument("--square-boxes", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32, help="Per-GPU batch size")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=60000)
    parser.add_argument("--val-samples", type=int, default=0)
    parser.add_argument("--max-gap", type=int, default=40)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--backbone-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--print-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--training-signature", default="")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank() if dist.is_initialized() else 0
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_set = UAVTrackingDataset(
        args.manifest, "train", args.samples_per_epoch, args.max_gap, square_boxes=args.square_boxes
    )
    val_set = None
    if args.val_samples > 0:
        val_set = UAVTrackingDataset(
            args.manifest, "val", args.val_samples, args.max_gap, square_boxes=args.square_boxes
        )
    train_sampler = DistributedSampler(train_set, shuffle=True) if world_size > 1 else None
    val_sampler = (
        DistributedSampler(val_set, shuffle=False)
        if world_size > 1 and val_set is not None
        else None
    )
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    train_loader = DataLoader(train_set, sampler=train_sampler, shuffle=train_sampler is None, **loader_args)
    val_loader = (
        DataLoader(val_set, sampler=val_sampler, shuffle=False, **loader_args)
        if val_set is not None
        else None
    )

    model = UAVTracker(
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        pretrained_path=args.pretrained_path,
        square_boxes=args.square_boxes,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(args.lr, args.backbone_lr_multiplier), weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    start_epoch, best_iou = 0, -1.0
    if args.resume and args.resume.exists():
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_iou = float(checkpoint.get("best_iou", -1.0))
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    if rank == 0:
        print(
            f"train trajectories={len(train_set.records)} "
            f"training_validation={'enabled' if val_set is not None else 'disabled'} "
            f"gpus={world_size}"
        )
        print(json.dumps(vars(args), default=str, ensure_ascii=False), flush=True)
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, epoch, True, args.print_interval)
        val_metrics = (
            run_epoch(model, val_loader, optimizer, scaler, device, epoch, False, 0)
            if val_loader is not None
            else None
        )
        scheduler.step()
        if rank == 0:
            current = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(current) + "\n")
            if val_metrics is None:
                print(f"epoch={epoch:03d} train_loss={train_metrics['total']:.4f}", flush=True)
                selection_score = -train_metrics["total"]
            else:
                print(
                    f"epoch={epoch:03d} train_loss={train_metrics['total']:.4f} "
                    f"val_iou={val_metrics['iou']:.4f}",
                    flush=True,
                )
                selection_score = val_metrics["iou"]
            improved = selection_score > best_iou
            if improved:
                best_iou = selection_score
            save_checkpoint(args.output_dir / "latest.pt", model, optimizer, scheduler, scaler, epoch, best_iou, args)
            if improved:
                save_checkpoint(args.output_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, best_iou, args)
                metric_name = "val_iou" if val_metrics is not None else "negative_train_loss"
                print(f"new_best epoch={epoch:03d} {metric_name}={best_iou:.4f}", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
