"""Pure queue/epoch checks: no subprocesses, models, or accelerators."""

import pytest

from experiments.toy_pairs.queue_intermediate import balance_groups, intermediate_epochs


def test_all_positive_intermediate_epochs_exclude_initial_and_final():
    mlp = [0, 10, *range(100, 1001, 100)]
    transformer = [*mlp, 1500, 2000, 2500]
    assert intermediate_epochs(mlp, 1000) == [10, *range(100, 1000, 100)]
    assert intermediate_epochs(transformer, 2500) == [10, *range(100, 1001, 100), 1500, 2000]
    assert 9 * len(intermediate_epochs(mlp, 1000)) + 3 * len(intermediate_epochs(transformer, 2500)) == 129
    with pytest.raises(ValueError):
        intermediate_epochs([0, 10, 10, 100], 100)
    with pytest.raises(ValueError):
        intermediate_epochs([0, 10], 100)


def test_group_assignment_preserves_all_seeds_and_sorts_each_gpu_by_epoch():
    groups = [{"dataset": dataset, "epoch": epoch, "cost_proxy": epoch + offset,
               "jobs": [{"seed": seed} for seed in (42, 43, 44)]}
              for dataset, offset in (("mnist", 0), ("modular_addition", 200))
              for epoch in (10, 100, 200, 300, 1000)]
    queues = balance_groups(groups, [1, 2, 3])
    assert 0 not in queues
    assert sorted((g["dataset"], g["epoch"]) for q in queues.values() for g in q) == sorted((g["dataset"], g["epoch"]) for g in groups)
    for queue in queues.values():
        assert [g["epoch"] for g in queue] == sorted(g["epoch"] for g in queue)
        assert all([job["seed"] for job in g["jobs"]] == [42, 43, 44] for g in queue)
