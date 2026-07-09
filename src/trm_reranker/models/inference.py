"""Architecture-agnostic single forward pass (including the ACT halting loop).

Shared by training, evaluation, and benchmarks so every consumer runs a model
the same way: models without ACT do one forward; recursive models with
``halt_max_steps > 1`` iterate the carry until every sequence halts.
"""

import torch

from ..utils import unwrap_model


def run_model_once(model, batch: dict[str, torch.Tensor]):
    stateful_model = unwrap_model(model)
    carry = stateful_model.initial_carry(batch)
    halt_max_steps = int(getattr(getattr(stateful_model, "config", None), "halt_max_steps", 1))

    if halt_max_steps == 1:
        carry, outputs = model(carry, batch)
        return outputs["scores"], outputs

    outputs = None
    for _ in range(halt_max_steps):
        carry, outputs = model(carry, batch)
        if bool(carry.halted.all()):
            break

    if outputs is None:
        raise RuntimeError("Model produced no outputs")

    return outputs["scores"], outputs
