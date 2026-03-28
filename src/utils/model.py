from dataclasses import dataclass

import torch.nn as nn


@dataclass(frozen=True)
class ParameterCounts:
    total: int
    trainable: int

    @property
    def non_trainable(self) -> int:
        return self.total - self.trainable


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def get_parameter_counts(model: nn.Module) -> ParameterCounts:
    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    return ParameterCounts(total=total, trainable=trainable)
