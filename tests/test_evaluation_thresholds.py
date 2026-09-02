import torch

from scripts.eval_generator_holdout import (
    eval_v as eval_generator_v,
    fit_threshold as fit_generator_threshold,
)
from scripts.eval_heldout_separability import (
    eval_v as eval_random_v,
    fit_threshold as fit_random_threshold,
)


def test_random_holdout_accuracy_uses_frozen_train_threshold():
    direction = torch.tensor([1.0])
    train_human = torch.tensor([[0.0], [0.0]])
    train_ai = torch.tensor([[2.0], [2.0]])
    test_human = torch.tensor([[100.0]])
    test_ai = torch.tensor([[102.0]])

    threshold = fit_random_threshold(
        direction,
        train_human,
        train_ai,
    )
    auc, accuracy = eval_random_v(
        direction,
        test_human,
        test_ai,
        threshold,
    )

    assert threshold == 1.0
    assert auc == 1.0
    assert accuracy == 0.5


def test_generator_holdout_accuracy_uses_frozen_train_threshold():
    direction = torch.tensor([1.0])
    train_human = torch.tensor([[-2.0], [-2.0]])
    train_ai = torch.tensor([[2.0], [2.0]])
    test_human = torch.tensor([[20.0]])
    test_ai = torch.tensor([[22.0]])

    threshold = fit_generator_threshold(
        direction,
        train_human,
        train_ai,
    )
    auc, accuracy = eval_generator_v(
        direction,
        test_human,
        test_ai,
        threshold,
    )

    assert threshold == 0.0
    assert auc == 1.0
    assert accuracy == 0.5
