"""Checkpoint save/load with the same payload keys as legacy notebooks."""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..utils import unwrap_model


@dataclass
class RunPaths:
    run_dir: Path
    checkpoint_dir: Path
    log_dir: Path
    eval_dir: Path
    epoch_eval_dir: Path
    step_eval_dir: Path
    final_eval_dir: Path
    train_log_path: Path
    training_config_path: Path
    fit_summary_path: Path
    run_artifacts_path: Path
    epoch_summaries_path: Path
    dev_metrics_by_epoch_path: Path
    dev_metrics_by_step_path: Path
    last_checkpoint_path: Path
    best_train_loss_checkpoint_path: Path
    best_mrr_checkpoint_path: Path
    best_dev_mrr10_checkpoint_path: Path
    final_run_path: Path
    final_metrics_path: Path

    @classmethod
    def create(cls, output_root: Path, experiment_name: str) -> "RunPaths":
        run_dir = Path(output_root).expanduser().resolve() / "runs" / experiment_name
        checkpoint_dir = run_dir / "checkpoints"
        log_dir = run_dir / "logs"
        eval_dir = run_dir / "eval"
        epoch_eval_dir = eval_dir / "epochs"
        step_eval_dir = eval_dir / "steps"
        final_eval_dir = eval_dir / "final"
        for directory in (run_dir, checkpoint_dir, log_dir, eval_dir, epoch_eval_dir, step_eval_dir, final_eval_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return cls(
            run_dir=run_dir,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            eval_dir=eval_dir,
            epoch_eval_dir=epoch_eval_dir,
            step_eval_dir=step_eval_dir,
            final_eval_dir=final_eval_dir,
            train_log_path=log_dir / "train_metrics.csv",
            training_config_path=run_dir / "training_config.json",
            fit_summary_path=run_dir / "fit_summary.json",
            run_artifacts_path=run_dir / "run_artifacts.json",
            epoch_summaries_path=log_dir / "epoch_summaries.json",
            dev_metrics_by_epoch_path=log_dir / "dev_metrics_by_epoch.csv",
            dev_metrics_by_step_path=log_dir / "dev_metrics_by_step.csv",
            last_checkpoint_path=checkpoint_dir / "last_checkpoint.pt",
            best_train_loss_checkpoint_path=checkpoint_dir / "best_train_loss.pt",
            best_mrr_checkpoint_path=checkpoint_dir / "best_mrr.pt",
            best_dev_mrr10_checkpoint_path=checkpoint_dir / "best_dev_mrr10.pt",
            final_run_path=final_eval_dir / "final_dev_best.run",
            final_metrics_path=final_eval_dir / "final_dev_metrics.json",
        )


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_step,
    config,
    metrics,
    steps_per_epoch: int = 1,
    target_total_steps: int = 0,
    run_steps_per_session: int = 0,
    checkpoint_every_steps=None,
):
    raw_model = unwrap_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = max(1, int(steps_per_epoch))
    completed_epochs = int(global_step // steps_per_epoch)
    step_in_epoch = int(global_step % steps_per_epoch)
    best_metric_value = metrics.get("best_mrr")
    if best_metric_value is None or not np.isfinite(float(best_metric_value)):
        best_metric_value = metrics.get("best_dev_mrr10")
    best_checkpoint_path = metrics.get("best_mrr_checkpoint_path") or metrics.get("best_dev_mrr10_checkpoint_path")
    torch.save(
        {
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "epoch": completed_epochs,
            "epoch_idx": int(epoch),
            "completed_epochs": completed_epochs,
            "step_in_epoch": step_in_epoch,
            "global_step": int(global_step),
            "estimated_steps_per_epoch": steps_per_epoch,
            "target_total_steps": int(target_total_steps),
            "run_steps_per_session": int(run_steps_per_session),
            "checkpoint_every_steps": checkpoint_every_steps,
            "checkpoint_time": time.time(),
            "best_metric": best_metric_value,
            "best_checkpoint_path": best_checkpoint_path,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def load_model_weights_for_eval(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    return model


def restore_training_state(model, optimizer, scheduler, scaler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint
