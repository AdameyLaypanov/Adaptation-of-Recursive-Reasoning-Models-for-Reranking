"""Training loop: faithful port of the legacy notebook trainer.

Same semantics as the notebooks: pairwise logistic loss, grad accumulation,
grad clipping, linear warmup+decay, EMA loss tracking, step/epoch dev evals,
best-train-loss / best-step-MRR / best-epoch-MRR / periodic checkpoints,
mid-epoch resume via ``ResumeDistributedSampler``, session windows through
``run_epoch_fraction`` / ``run_train_steps``.
"""

import csv
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from ..config import TrainingSection
from ..data.datasets import ResumeDistributedSampler
from ..data.encoding import PairEncoder, move_batch_to_device
from ..evaluation.metrics import RANKING_METRIC_NAMES
from ..utils import make_tqdm, save_json
from .checkpoints import RunPaths, restore_training_state, save_checkpoint
from .distributed import (
    ddp_barrier,
    get_autocast_context,
    get_local_rank,
    get_rank,
    is_dist_available_and_initialized,
    is_main_process,
    reduce_mean,
    unwrap_model,
)
from .optim import (
    build_linear_warmup_decay_lambda,
    build_warmup_steps,
    compute_pairwise_batch_metrics,
    count_model_parameters,
    count_trainable_parameters,
    get_trainable_parameters,
    resolve_step_count,
)

TRAIN_LOG_FIELDNAMES = [
    "time",
    "epoch",
    "batch_idx",
    "global_step",
    "loss",
    "loss_ema",
    "pairwise_acc",
    "margin",
    "pos_score",
    "neg_score",
    "lr",
    "grad_norm",
]
DEV_METRICS_FIELDNAMES = [
    "epoch",
    "global_step",
    "dev_eval_mode",
    "queries_evaluated",
    *[f"bm25_{name}" for name in RANKING_METRIC_NAMES],
    *[f"trm_{name}" for name in RANKING_METRIC_NAMES],
    "run_path",
    "metrics_path",
]
DEV_METRICS_BY_STEP_FIELDNAMES = DEV_METRICS_FIELDNAMES + [
    "dev_eval_query_limit",
    "dev_eval_every_steps",
    "best_mrr",
    "best_mrr_updated",
    "best_mrr_checkpoint_path",
]


def save_csv_rows(path: Path, fieldnames, rows) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def create_train_log_writer(path: Path, append: bool = False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    should_append = append and path.exists() and path.stat().st_size > 0
    handle = path.open("a" if should_append else "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=TRAIN_LOG_FIELDNAMES)
    if not should_append:
        writer.writeheader()
    writer._handle = handle
    return writer


def close_train_log_writer(writer) -> None:
    if writer is not None and hasattr(writer, "_handle"):
        writer._handle.close()


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        cfg: TrainingSection,
        paths: RunPaths,
        train_dataset,
        pairwise_collate,
        encoder: PairEncoder,
        dev_query_tokens: Dict,
        epoch_dev_candidates,
        epoch_dev_qrels,
        passage_token_getter,
        seed: int,
        effective_precision: str,
        should_use_ddp: bool,
        world_size: int,
        epoch_dev_mode_label: str = "quick",
        final_dev_candidates=None,
        final_dev_qrels=None,
        run_final_full_dev: bool = False,
        extra_training_summary: Optional[Dict] = None,
    ):
        self.cfg = cfg
        self.paths = paths
        self.device = device
        self.encoder = encoder
        self.train_dataset = train_dataset
        self.pairwise_collate = pairwise_collate
        self.dev_query_tokens = dev_query_tokens
        self.epoch_dev_candidates = epoch_dev_candidates
        self.epoch_dev_qrels = epoch_dev_qrels
        self.passage_token_getter = passage_token_getter
        self.seed = int(seed)
        self.precision = effective_precision
        self.should_use_ddp = bool(should_use_ddp)
        self.world_size = int(world_size)
        self.epoch_dev_mode_label = epoch_dev_mode_label
        self.final_dev_candidates = final_dev_candidates
        self.final_dev_qrels = final_dev_qrels
        self.run_final_full_dev = bool(run_final_full_dev)

        self.model = model.to(device)
        self.total_parameters = count_model_parameters(self.model)
        self.freeze_diagnostics = self._apply_embedding_freeze()
        self.trainable_parameters = count_trainable_parameters(self.model)
        if self.should_use_ddp:
            self.model = DDP(self.model, device_ids=[get_local_rank()], output_device=get_local_rank(), find_unused_parameters=False)

        optimizer_parameters = get_trainable_parameters(self.model)
        self.optimizer = torch.optim.AdamW(optimizer_parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        # Step accounting mirrors the notebooks.
        probe_loader, _ = self._build_train_loader(epoch=0, start_batch_idx=0)
        self.steps_per_epoch = max(1, math.ceil(len(probe_loader) / cfg.grad_accum_steps))
        self.target_total_steps = self.steps_per_epoch * cfg.epochs
        self.run_steps_per_session = min(
            int(
                resolve_step_count(
                    cfg.run_train_steps,
                    cfg.run_epoch_fraction,
                    self.steps_per_epoch,
                    default_steps=self.steps_per_epoch,
                )
            ),
            int(self.target_total_steps),
        )
        checkpoint_every = resolve_step_count(
            cfg.checkpoint_every_n_steps,
            cfg.checkpoint_epoch_fraction,
            self.steps_per_epoch,
            default_steps=self.run_steps_per_session,
        )
        self.checkpoint_every_steps = int(checkpoint_every) if checkpoint_every is not None else None
        self.dev_eval_every_steps = self.checkpoint_every_steps

        warmup_steps = build_warmup_steps(self.target_total_steps, cfg.warmup_ratio)
        self.warmup_steps = warmup_steps
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=build_linear_warmup_decay_lambda(warmup_steps, self.target_total_steps)
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and effective_precision == "16-mixed")

        self.training_summary = {
            "num_train_examples": len(train_dataset),
            "epochs": cfg.epochs,
            "max_train_steps": cfg.max_train_steps,
            "per_device_batch_size": cfg.per_device_batch_size,
            "world_size": self.world_size,
            "grad_accum_steps": cfg.grad_accum_steps,
            "global_batch_size": cfg.per_device_batch_size * self.world_size * cfg.grad_accum_steps,
            "estimated_steps_per_epoch": self.steps_per_epoch,
            "estimated_total_steps": self.target_total_steps,
            "target_total_steps": self.target_total_steps,
            "run_steps_per_session": self.run_steps_per_session,
            "checkpoint_every_steps": self.checkpoint_every_steps,
            "dev_eval_query_limit": cfg.dev_eval_query_limit,
            "dev_eval_every_steps": self.dev_eval_every_steps,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "warmup_steps": warmup_steps,
            "max_grad_norm": cfg.max_grad_norm,
            "precision": effective_precision,
            "seed": self.seed,
            "run_dir": str(paths.run_dir),
            "resume_from_checkpoint": cfg.resume_from_checkpoint,
            "resume_checkpoint_path": cfg.resume_checkpoint_path,
            "ddp_enabled": self.should_use_ddp,
            "freeze_token_embeddings": cfg.freeze_token_embeddings,
            "freeze_segment_embeddings": cfg.freeze_segment_embeddings,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            **self.freeze_diagnostics,
        }
        if extra_training_summary:
            self.training_summary.update(extra_training_summary)

        # Mutable best-run state.
        self.global_step = 0
        self.best_train_loss = float("inf")
        self.best_train_loss_step = None
        self.best_mrr = float("-inf")
        self.best_dev_mrr10 = float("-inf")
        self.best_mrr_checkpoint_path = None
        self.best_train_loss_checkpoint_path = None
        self.best_dev_mrr10_checkpoint_path = None
        self.loss_ema = None
        self.epoch_summaries = []
        self.dev_metrics_history = []
        self.step_dev_metrics_history = []

    # ------------------------------------------------------------------ setup

    def _apply_embedding_freeze(self) -> Dict:
        diagnostics = {"token_embeddings_frozen": False, "segment_embeddings_frozen": False}
        raw_model = unwrap_model(self.model)
        inner = getattr(raw_model, "inner", None)
        if self.cfg.freeze_token_embeddings:
            if inner is None or not hasattr(inner, "embed_tokens"):
                raise AttributeError("freeze_token_embeddings=True, but model has no inner.embed_tokens")
            inner.embed_tokens.embedding_weight.requires_grad_(False)
            diagnostics["token_embeddings_frozen"] = True
        if self.cfg.freeze_segment_embeddings:
            if inner is None or not hasattr(inner, "segment_emb"):
                raise AttributeError("freeze_segment_embeddings=True, but model has no inner.segment_emb")
            inner.segment_emb.embedding_weight.requires_grad_(False)
            diagnostics["segment_embeddings_frozen"] = True
        return diagnostics

    def _build_train_loader(self, epoch: int, start_batch_idx: int = 0):
        start_index = max(0, int(start_batch_idx)) * int(self.cfg.per_device_batch_size)
        sampler = ResumeDistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=get_rank() if self.should_use_ddp else 0,
            shuffle=True,
            seed=self.seed,
            drop_last=False,
            start_index=start_index,
        )
        sampler.set_epoch(int(epoch))
        loader_kwargs = dict(
            batch_size=self.cfg.per_device_batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=0,
            collate_fn=self.pairwise_collate,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        loader = DataLoader(self.train_dataset, **loader_kwargs)
        return loader, sampler

    # ------------------------------------------------------------- checkpoints

    def _active_scaler(self):
        return self.scaler if self.scaler.is_enabled() else None

    def _make_checkpoint_metrics(self, epoch_summary=None, step_row=None, loss_ema_value=None, dev_metrics=None, dev_mode_label=None):
        return {
            "epoch_summary": epoch_summary,
            "step_row": step_row,
            "best_train_loss": self.best_train_loss,
            "best_train_loss_step": self.best_train_loss_step,
            "best_mrr": self.best_mrr,
            "best_mrr_metric_name": "trm_mrr@10",
            "best_mrr_checkpoint_path": self.best_mrr_checkpoint_path,
            "best_dev_mrr10": self.best_dev_mrr10,
            "best_train_loss_checkpoint_path": self.best_train_loss_checkpoint_path,
            "best_dev_mrr10_checkpoint_path": self.best_dev_mrr10_checkpoint_path,
            "loss_ema": self.loss_ema if loss_ema_value is None else loss_ema_value,
            "dev_eval_metrics": dev_metrics,
            "dev_eval_mode": dev_mode_label,
            "target_total_steps": self.target_total_steps,
            "estimated_steps_per_epoch": self.steps_per_epoch,
            "dev_eval_query_limit": self.cfg.dev_eval_query_limit,
            "dev_eval_every_steps": self.dev_eval_every_steps,
        }

    def _save_checkpoint(self, path, epoch, metrics):
        save_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.scheduler,
            self._active_scaler(),
            epoch,
            self.global_step,
            self.training_summary,
            metrics,
            steps_per_epoch=self.steps_per_epoch,
            target_total_steps=self.target_total_steps,
            run_steps_per_session=self.run_steps_per_session,
            checkpoint_every_steps=self.checkpoint_every_steps,
        )

    # -------------------------------------------------------------- dev evals

    def _evaluate(self, candidates, qrels, run_path, query_limit=None, per_query_path=None):
        from ..evaluation.reranker_eval import evaluate_reranker  # lazy: breaks the trainer <-> eval import cycle

        return evaluate_reranker(
            unwrap_model(self.model),
            self.encoder,
            candidates,
            qrels,
            self.dev_query_tokens,
            run_path=run_path,
            passage_token_getter=self.passage_token_getter,
            eval_batch_size=self.cfg.eval_batch_size,
            precision=self.precision,
            query_limit=query_limit,
            per_query_path=per_query_path,
        )

    def _run_step_dev_eval(self, epoch: int):
        metrics = None
        query_limit = self.cfg.dev_eval_query_limit
        step_dev_mode_label = f"step_query_limit_{int(query_limit)}"
        if is_main_process():
            step_name = f"dev_step_{self.global_step:08d}_qlimit_{int(query_limit)}"
            run_path = self.paths.step_eval_dir / f"{step_name}.run"
            metrics_path = self.paths.step_eval_dir / f"{step_name}_metrics.json"
            with torch.no_grad():
                metrics = self._evaluate(self.epoch_dev_candidates, self.epoch_dev_qrels, run_path, query_limit=query_limit)
            metrics["epoch"] = epoch + 1
            metrics["global_step"] = self.global_step
            metrics["dev_eval_mode"] = step_dev_mode_label
            metrics["dev_eval_query_limit"] = int(query_limit)
            metrics["dev_eval_every_steps"] = self.dev_eval_every_steps
            metrics["metrics_path"] = str(metrics_path)
            save_json(metrics_path, metrics)
        return metrics, step_dev_mode_label

    def _run_epoch_dev_eval(self, epoch: int):
        metrics = None
        if is_main_process():
            epoch_name = f"dev_epoch_{epoch + 1:03d}_{self.epoch_dev_mode_label}"
            run_path = self.paths.epoch_eval_dir / f"{epoch_name}.run"
            metrics_path = self.paths.epoch_eval_dir / f"{epoch_name}_metrics.json"
            per_query_path = self.paths.epoch_eval_dir / f"{epoch_name}_per_query.csv"
            with torch.no_grad():
                metrics = self._evaluate(self.epoch_dev_candidates, self.epoch_dev_qrels, run_path, per_query_path=per_query_path)
            metrics["epoch"] = epoch + 1
            metrics["global_step"] = self.global_step
            metrics["dev_eval_mode"] = self.epoch_dev_mode_label
            metrics["metrics_path"] = str(metrics_path)
            save_json(metrics_path, metrics)
        return metrics, self.epoch_dev_mode_label

    def _select_final_eval_checkpoint(self) -> Optional[Path]:
        candidates = [
            self.paths.best_mrr_checkpoint_path,
            self.best_mrr_checkpoint_path,
            self.best_dev_mrr10_checkpoint_path,
            self.paths.best_dev_mrr10_checkpoint_path,
            self.best_train_loss_checkpoint_path,
            self.paths.best_train_loss_checkpoint_path,
            self.paths.last_checkpoint_path,
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            candidate = Path(candidate)
            if candidate.exists():
                return candidate
        return None

    def run_final_full_dev_eval(self):
        from .checkpoints import load_model_weights_for_eval

        checkpoint_path = self._select_final_eval_checkpoint()
        if checkpoint_path is None:
            raise FileNotFoundError("No checkpoint is available for final full dev evaluation.")
        metrics = None
        if is_main_process():
            eval_model = unwrap_model(self.model)
            load_model_weights_for_eval(eval_model, checkpoint_path, self.device)
            per_query_path = self.paths.final_eval_dir / "final_dev_per_query.csv"
            with torch.no_grad():
                metrics = self._evaluate(
                    self.final_dev_candidates,
                    self.final_dev_qrels,
                    self.paths.final_run_path,
                    per_query_path=per_query_path,
                )
            metrics["dev_eval_mode"] = "final_full"
            metrics["checkpoint_path"] = str(checkpoint_path)
            metrics["metrics_path"] = str(self.paths.final_metrics_path)
            save_json(self.paths.final_metrics_path, metrics)
        ddp_barrier()
        return metrics, checkpoint_path

    # ---------------------------------------------------------------- resume

    def _maybe_resume(self):
        if not self.cfg.resume_from_checkpoint:
            return
        resume_path = Path(self.cfg.resume_checkpoint_path) if self.cfg.resume_checkpoint_path else self.paths.last_checkpoint_path
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        if is_main_process():
            print(f"Loading resume checkpoint: {resume_path}")
        checkpoint = restore_training_state(self.model, self.optimizer, self.scheduler, self._active_scaler(), resume_path, self.device)
        self.global_step = int(checkpoint["global_step"])
        checkpoint_steps_per_epoch = checkpoint.get("estimated_steps_per_epoch")
        if checkpoint_steps_per_epoch is not None and int(checkpoint_steps_per_epoch) != int(self.steps_per_epoch):
            raise ValueError(
                f"The checkpoint was created with estimated_steps_per_epoch={checkpoint_steps_per_epoch}, "
                f"but the current run has {self.steps_per_epoch}. Do not change train sample count, batch size, "
                "world size, or grad_accum_steps when resuming mid-epoch."
            )
        resume_metrics = checkpoint.get("metrics") or {}
        self.best_train_loss = float(resume_metrics.get("best_train_loss", self.best_train_loss))
        best_train_loss_step = resume_metrics.get("best_train_loss_step")
        self.best_train_loss_step = int(best_train_loss_step) if best_train_loss_step is not None else None
        best_mrr = (
            resume_metrics.get("best_mrr")
            or resume_metrics.get("best_dev_mrr10")
            or checkpoint.get("best_metric")
            or resume_metrics.get("best_quick_mrr10", self.best_mrr)
        )
        self.best_mrr = float(best_mrr)
        best_dev_mrr10 = (
            resume_metrics.get("best_dev_mrr10")
            or resume_metrics.get("best_full_dev_mrr10")
            or checkpoint.get("best_metric")
            or resume_metrics.get("best_quick_mrr10", self.best_dev_mrr10)
        )
        self.best_dev_mrr10 = float(best_dev_mrr10)
        self.best_train_loss_checkpoint_path = resume_metrics.get("best_train_loss_checkpoint_path")
        self.best_mrr_checkpoint_path = (
            resume_metrics.get("best_mrr_checkpoint_path")
            or resume_metrics.get("best_dev_mrr10_checkpoint_path")
            or checkpoint.get("best_checkpoint_path")
        )
        self.best_dev_mrr10_checkpoint_path = resume_metrics.get("best_dev_mrr10_checkpoint_path") or self.best_mrr_checkpoint_path
        self.loss_ema = resume_metrics.get("loss_ema")
        if self.cfg.allow_resume_lr_override:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.cfg.learning_rate
                param_group["initial_lr"] = self.cfg.learning_rate
            self.scheduler.base_lrs = [self.cfg.learning_rate for _ in self.optimizer.param_groups]
        if is_main_process():
            print(f"Resuming from global_step={self.global_step}")

    # ------------------------------------------------------------- inner loop

    def _train_one_epoch(self, train_loader, epoch, target_global_step, batch_idx_offset, log_writer):
        cfg = self.cfg
        model = self.model
        model.train()
        self.optimizer.zero_grad(set_to_none=True)
        progress_desc = f"Epoch {epoch + 1}/{cfg.epochs}"
        if batch_idx_offset > 0:
            progress_desc += f" | resume batch {batch_idx_offset}"
        progress_bar = make_tqdm(enumerate(train_loader), total=len(train_loader), disable=not is_main_process(), desc=progress_desc)
        micro_accumulator = {key: [] for key in ["loss", "pairwise_acc", "margin", "pos_score", "neg_score"]}
        step_rows = []
        stop_training = False
        scaler = self._active_scaler()

        for batch_idx, (pos_batch, neg_batch) in progress_bar:
            absolute_batch_idx = batch_idx_offset + batch_idx
            pos_batch = move_batch_to_device(pos_batch, self.device)
            neg_batch = move_batch_to_device(neg_batch, self.device)
            is_last_batch = batch_idx + 1 == len(train_loader)
            should_step = ((absolute_batch_idx + 1) % cfg.grad_accum_steps == 0) or is_last_batch
            sync_context = nullcontext()
            if is_dist_available_and_initialized() and hasattr(model, "no_sync") and not should_step:
                sync_context = model.no_sync()

            with sync_context:
                with get_autocast_context(self.device, self.precision):
                    batch_metrics = compute_pairwise_batch_metrics(model, pos_batch, neg_batch)
                    loss = batch_metrics["loss"]
                    loss_for_backward = loss / cfg.grad_accum_steps
                if scaler is not None:
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

            for key in micro_accumulator:
                micro_accumulator[key].append(batch_metrics[key].detach())

            if not should_step:
                continue

            if scaler is not None:
                scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), cfg.max_grad_norm)
            if scaler is not None:
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

            reduced_metrics = {}
            for key, values in micro_accumulator.items():
                stacked = torch.stack(values)
                reduced_metrics[key] = float(reduce_mean(stacked.mean()).item())
            grad_norm_value = float(reduce_mean(torch.as_tensor(float(grad_norm), device=self.device)).item())
            lr_value = float(reduce_mean(torch.as_tensor(float(self.optimizer.param_groups[0]["lr"]), device=self.device)).item())
            self.loss_ema = (
                reduced_metrics["loss"] if self.loss_ema is None else (0.98 * self.loss_ema + 0.02 * reduced_metrics["loss"])
            )

            row = {
                "time": time.time(),
                "epoch": epoch + 1,
                "batch_idx": absolute_batch_idx,
                "global_step": self.global_step,
                "loss": reduced_metrics["loss"],
                "loss_ema": self.loss_ema,
                "pairwise_acc": reduced_metrics["pairwise_acc"],
                "margin": reduced_metrics["margin"],
                "pos_score": reduced_metrics["pos_score"],
                "neg_score": reduced_metrics["neg_score"],
                "lr": lr_value,
                "grad_norm": grad_norm_value,
            }
            step_rows.append(row)

            if log_writer is not None and is_main_process():
                log_writer.writerow(row)
                log_writer._handle.flush()

            if is_main_process() and self.global_step % cfg.tqdm_postfix_every_n_steps == 0:
                progress_bar.set_postfix(
                    {
                        "global_step": self.global_step,
                        "loss": f"{row['loss']:.4f}",
                        "loss_ema": f"{row['loss_ema']:.4f}",
                        "acc": f"{row['pairwise_acc']:.3f}",
                        "lr": f"{row['lr']:.2e}",
                    },
                    refresh=False,
                )

            self._on_best_train_loss(epoch, row)

            should_checkpoint = (
                self.checkpoint_every_steps is not None
                and self.checkpoint_every_steps > 0
                and self.global_step % self.checkpoint_every_steps == 0
            ) or (target_global_step is not None and self.global_step >= target_global_step)
            if should_checkpoint:
                self._on_periodic_checkpoint(epoch, row)

            should_run_dev_eval = (
                self.dev_eval_every_steps is not None
                and self.dev_eval_every_steps > 0
                and self.global_step % self.dev_eval_every_steps == 0
            ) or (target_global_step is not None and self.global_step == target_global_step)
            if should_run_dev_eval:
                self._on_step_dev_eval(epoch, row)

            micro_accumulator = {key: [] for key in micro_accumulator}
            if target_global_step is not None and self.global_step >= target_global_step:
                stop_training = True
                break
            if cfg.max_train_steps is not None and self.global_step >= int(cfg.max_train_steps):
                stop_training = True
                break

        if is_main_process():
            progress_bar.close()

        if step_rows:
            epoch_summary = {
                "epoch": epoch + 1,
                "epoch_idx": epoch,
                "global_step": self.global_step,
                "start_batch_idx": batch_idx_offset,
                "end_batch_idx": int(step_rows[-1]["batch_idx"]),
                "target_global_step": target_global_step,
                "optimizer_steps": len(step_rows),
                "loss": float(np.mean([row["loss"] for row in step_rows])),
                "loss_ema": float(step_rows[-1]["loss_ema"]),
                "pairwise_acc": float(np.mean([row["pairwise_acc"] for row in step_rows])),
                "margin": float(np.mean([row["margin"] for row in step_rows])),
                "pos_score": float(np.mean([row["pos_score"] for row in step_rows])),
                "neg_score": float(np.mean([row["neg_score"] for row in step_rows])),
                "lr": float(step_rows[-1]["lr"]),
                "grad_norm": float(np.mean([row["grad_norm"] for row in step_rows])),
                "epoch_completed": bool(step_rows[-1]["batch_idx"] + 1 >= batch_idx_offset + len(train_loader)),
            }
        else:
            epoch_summary = {
                "epoch": epoch + 1,
                "epoch_idx": epoch,
                "global_step": self.global_step,
                "start_batch_idx": batch_idx_offset,
                "end_batch_idx": None,
                "target_global_step": target_global_step,
                "optimizer_steps": 0,
                "loss": None,
                "loss_ema": self.loss_ema,
                "pairwise_acc": None,
                "margin": None,
                "pos_score": None,
                "neg_score": None,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "grad_norm": None,
                "epoch_completed": False,
            }
        return epoch_summary, stop_training

    # ---------------------------------------------------------------- callbacks

    def _on_best_train_loss(self, epoch, row):
        step_loss = row.get("loss")
        if step_loss is None:
            return
        step_loss = float(step_loss)
        if not np.isfinite(step_loss) or step_loss >= self.best_train_loss:
            return
        self.best_train_loss = step_loss
        self.best_train_loss_step = int(self.global_step)
        self.best_train_loss_checkpoint_path = str(self.paths.best_train_loss_checkpoint_path)
        if not is_main_process():
            return
        metrics = self._make_checkpoint_metrics(step_row=row, loss_ema_value=self.loss_ema, dev_mode_label="not_run_step_best_train_loss")
        self._save_checkpoint(self.paths.best_train_loss_checkpoint_path, epoch, metrics)

    def _on_periodic_checkpoint(self, epoch, row):
        if not is_main_process():
            return
        metrics = self._make_checkpoint_metrics(step_row=row, loss_ema_value=self.loss_ema, dev_mode_label="not_run_step_checkpoint")
        step_checkpoint_path = self.paths.checkpoint_dir / f"step_{self.global_step:08d}.pt"
        self._save_checkpoint(step_checkpoint_path, epoch, metrics)
        self._save_checkpoint(self.paths.last_checkpoint_path, epoch, metrics)

    def _on_step_dev_eval(self, epoch, row):
        if is_main_process():
            dev_metrics, dev_mode_label = self._run_step_dev_eval(epoch)
            best_mrr_updated = self._maybe_update_best_mrr(epoch, row, dev_metrics)
            if dev_metrics is not None:
                dev_metrics_row = self._build_dev_metrics_row(dev_metrics, epoch + 1, dev_mode_label)
                dev_metrics_row.update(
                    {
                        "dev_eval_query_limit": self.cfg.dev_eval_query_limit,
                        "dev_eval_every_steps": self.dev_eval_every_steps,
                        "best_mrr": self.best_mrr,
                        "best_mrr_updated": bool(best_mrr_updated),
                        "best_mrr_checkpoint_path": self.best_mrr_checkpoint_path,
                    }
                )
                self.step_dev_metrics_history.append(dev_metrics_row)
                save_csv_rows(self.paths.dev_metrics_by_step_path, DEV_METRICS_BY_STEP_FIELDNAMES, self.step_dev_metrics_history)
        ddp_barrier()

    def _maybe_update_best_mrr(self, epoch, step_row, dev_metrics) -> bool:
        if dev_metrics is None or "trm_mrr@10" not in dev_metrics:
            return False
        step_mrr = float(dev_metrics["trm_mrr@10"])
        if not np.isfinite(step_mrr) or step_mrr <= self.best_mrr:
            return False
        self.best_mrr = step_mrr
        self.best_mrr_checkpoint_path = str(self.paths.best_mrr_checkpoint_path)
        metrics = self._make_checkpoint_metrics(
            step_row=step_row, loss_ema_value=self.loss_ema, dev_metrics=dev_metrics, dev_mode_label=dev_metrics.get("dev_eval_mode")
        )
        metrics.update({"best_mrr_updated": True, "global_step": int(self.global_step), "epoch": int(epoch) + 1})
        self._save_checkpoint(self.paths.best_mrr_checkpoint_path, epoch, metrics)
        return True

    def _build_dev_metrics_row(self, metrics, epoch, dev_eval_mode):
        row = {
            "epoch": epoch,
            "global_step": self.global_step,
            "dev_eval_mode": dev_eval_mode,
            "queries_evaluated": int(metrics["queries_evaluated"]),
            "run_path": str(metrics["run_path"]),
            "metrics_path": str(metrics.get("metrics_path", "")),
        }
        for prefix in ("bm25", "trm"):
            for metric_name in RANKING_METRIC_NAMES:
                row[f"{prefix}_{metric_name}"] = float(metrics[f"{prefix}_{metric_name}"])
        return row

    # -------------------------------------------------------------------- fit

    def fit(self):
        cfg = self.cfg
        self._maybe_resume()
        ddp_barrier()

        if is_main_process():
            save_json(self.paths.training_config_path, self.training_summary)

        log_writer = create_train_log_writer(self.paths.train_log_path, append=cfg.resume_from_checkpoint) if is_main_process() else None
        run_start_global_step = int(self.global_step)
        run_target_global_step = min(int(self.target_total_steps), run_start_global_step + int(self.run_steps_per_session))
        final_dev_metrics = None
        final_eval_checkpoint_path = None
        stop_training = False

        if is_main_process():
            print(
                f"Step-based training window: global_step {run_start_global_step} -> {run_target_global_step} / {self.target_total_steps}"
            )

        try:
            while self.global_step < int(self.target_total_steps) and self.global_step < int(run_target_global_step):
                epoch = int(self.global_step // self.steps_per_epoch)
                step_in_epoch = int(self.global_step % self.steps_per_epoch)
                probe_loader_len = self.steps_per_epoch * cfg.grad_accum_steps
                start_batch_idx = min(probe_loader_len, step_in_epoch * cfg.grad_accum_steps)
                train_loader, _ = self._build_train_loader(epoch=epoch, start_batch_idx=start_batch_idx)

                epoch_summary, stop_training = self._train_one_epoch(
                    train_loader,
                    epoch=epoch,
                    target_global_step=run_target_global_step,
                    batch_idx_offset=start_batch_idx,
                    log_writer=log_writer,
                )
                epoch_summary["run_start_global_step"] = run_start_global_step
                epoch_summary["run_target_global_step"] = run_target_global_step
                epoch_summary["target_total_steps"] = self.target_total_steps
                epoch_summary["completed_epochs_float"] = float(self.global_step / self.steps_per_epoch)

                completed_epoch_boundary = self.global_step > 0 and self.global_step % self.steps_per_epoch == 0
                reached_total_target = self.global_step >= int(self.target_total_steps)
                should_run_epoch_eval = completed_epoch_boundary or reached_total_target or cfg.run_epoch_dev_eval_on_partial

                epoch_dev_metrics = None
                epoch_dev_mode_label = "skipped_partial_step_chunk"
                if should_run_epoch_eval:
                    epoch_dev_metrics, epoch_dev_mode_label = self._run_epoch_dev_eval(epoch)
                epoch_summary["dev_eval_mode"] = epoch_dev_mode_label
                if epoch_dev_metrics is not None:
                    epoch_summary["dev_mrr10"] = float(epoch_dev_metrics["trm_mrr@10"])
                    epoch_summary["dev_bm25_mrr10"] = float(epoch_dev_metrics["bm25_mrr@10"])
                    epoch_summary["dev_queries_evaluated"] = int(epoch_dev_metrics["queries_evaluated"])
                    self.dev_metrics_history.append(self._build_dev_metrics_row(epoch_dev_metrics, epoch + 1, epoch_dev_mode_label))
                self.epoch_summaries.append(epoch_summary)

                improved_dev_mrr10 = epoch_dev_metrics is not None and float(epoch_dev_metrics["trm_mrr@10"]) > self.best_dev_mrr10
                if improved_dev_mrr10:
                    self.best_dev_mrr10 = float(epoch_dev_metrics["trm_mrr@10"])
                    self.best_dev_mrr10_checkpoint_path = str(self.paths.best_dev_mrr10_checkpoint_path)

                if is_main_process():
                    save_json(self.paths.epoch_summaries_path, self.epoch_summaries)
                    save_csv_rows(self.paths.dev_metrics_by_epoch_path, DEV_METRICS_FIELDNAMES, self.dev_metrics_history)
                    checkpoint_metrics = self._make_checkpoint_metrics(
                        epoch_summary=epoch_summary,
                        loss_ema_value=self.loss_ema,
                        dev_metrics=epoch_dev_metrics,
                        dev_mode_label=epoch_dev_mode_label,
                    )
                    if completed_epoch_boundary:
                        segment_checkpoint_path = self.paths.checkpoint_dir / f"epoch_{self.global_step // self.steps_per_epoch:03d}.pt"
                    else:
                        segment_checkpoint_path = self.paths.checkpoint_dir / f"step_{self.global_step:08d}.pt"
                    self._save_checkpoint(segment_checkpoint_path, epoch, checkpoint_metrics)
                    self._save_checkpoint(self.paths.last_checkpoint_path, epoch, checkpoint_metrics)
                    if improved_dev_mrr10:
                        self._save_checkpoint(self.paths.best_dev_mrr10_checkpoint_path, epoch, checkpoint_metrics)
                ddp_barrier()
                if epoch_summary.get("optimizer_steps", 0) == 0:
                    if is_main_process():
                        print("No optimizer steps were completed in this segment; stopping to avoid an infinite loop.")
                    break
                if stop_training:
                    break
        finally:
            close_train_log_writer(log_writer)

        if self.run_final_full_dev and self.global_step >= int(self.target_total_steps):
            final_dev_metrics, final_eval_checkpoint_path = self.run_final_full_dev_eval()
        elif self.run_final_full_dev and is_main_process():
            print(
                f"Skipping final full dev eval until training reaches target_total_steps: "
                f"global_step={self.global_step}, target_total_steps={self.target_total_steps}"
            )

        fit_summary = {
            "global_step": int(self.global_step),
            "training_complete": bool(self.global_step >= int(self.target_total_steps)),
            "epochs_completed": int(self.global_step // self.steps_per_epoch),
            "epochs_completed_float": float(self.global_step / self.steps_per_epoch),
            "segments_completed_this_run": len(self.epoch_summaries),
            "target_total_steps": int(self.target_total_steps),
            "run_start_global_step": run_start_global_step,
            "run_target_global_step": run_target_global_step,
            "run_steps_per_session": int(self.run_steps_per_session),
            "checkpoint_every_steps": self.checkpoint_every_steps,
            "dev_eval_query_limit": int(cfg.dev_eval_query_limit),
            "dev_eval_every_steps": self.dev_eval_every_steps,
            "checkpoint_dir": str(self.paths.checkpoint_dir),
            "log_dir": str(self.paths.log_dir),
            "train_log_path": str(self.paths.train_log_path),
            "last_checkpoint_path": str(self.paths.last_checkpoint_path) if self.paths.last_checkpoint_path.exists() else None,
            "best_mrr": None if not np.isfinite(self.best_mrr) else float(self.best_mrr),
            "best_mrr_checkpoint_path": str(self.paths.best_mrr_checkpoint_path) if self.paths.best_mrr_checkpoint_path.exists() else None,
            "best_train_loss": None if not np.isfinite(self.best_train_loss) else float(self.best_train_loss),
            "best_train_loss_step": self.best_train_loss_step,
            "best_train_loss_checkpoint_path": (
                str(self.paths.best_train_loss_checkpoint_path) if self.paths.best_train_loss_checkpoint_path.exists() else None
            ),
            "best_dev_mrr10_checkpoint_path": (
                str(self.paths.best_dev_mrr10_checkpoint_path) if self.paths.best_dev_mrr10_checkpoint_path.exists() else None
            ),
            "final_eval_checkpoint_path": str(final_eval_checkpoint_path) if final_eval_checkpoint_path is not None else None,
            "final_metrics_path": str(self.paths.final_metrics_path) if self.paths.final_metrics_path.exists() else None,
        }
        if final_dev_metrics is not None:
            fit_summary["final_full_dev_mrr10"] = float(final_dev_metrics["trm_mrr@10"])
            fit_summary["final_full_dev_bm25_mrr10"] = float(final_dev_metrics["bm25_mrr@10"])
            fit_summary["final_full_dev_queries_evaluated"] = int(final_dev_metrics["queries_evaluated"])
        if is_main_process():
            save_json(self.paths.fit_summary_path, fit_summary)
            print(f"fit_summary saved to {self.paths.fit_summary_path}")
        return fit_summary
