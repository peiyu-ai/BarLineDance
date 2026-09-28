"""What a multi-GPU run must still be, tested on CPU with gloo.

Two distinct claims live in ``train_atomic``'s data-parallel path, and each has a
failure mode that produces a run which looks entirely normal:

1. ``resolve_batch_sizes``' promise that ``--global-batch-size`` keeps the
   single-GPU recipe: DDP averages the per-rank gradients, and with equal rank
   batches that average *is* the gradient over the whole batch.  If it were not,
   a run launched to stay comparable with an existing arm would quietly not be.

2. The diffusion wrappers' ``forward`` alias.  DDP hangs its reduction hooks off
   ``forward``; reaching past it to ``training_step`` runs the same arithmetic
   with no synchronisation, so each rank trains its own copy on its own shard.
   The loss curve looks ordinary and the checkpoint is one that saw 1/N of the
   corpus.  The second test asserts the reduction fires -- and, on the same
   fixture, that skipping it is detectable, so the assertion is not vacuous.

These run on gloo/CPU deliberately: the property is DDP's, not the card's, and a
GPU-only test would be skipped exactly where it matters least.
"""

import os
import tempfile
import unittest

import torch
import torch.multiprocessing
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from model.atomic_planner import AtomicPlannerTransformer, UniformD3PM

WORLD_SIZE = 2
BATCH = 8
FRAMES = 6
CLASSES = 5


class _Quadratic(nn.Module):
    """A model whose gradient can be written down, so the test has an oracle."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2, bias=False)

    def forward(self, inputs, targets):
        return ((self.linear(inputs) - targets) ** 2).mean()


def _fixture():
    generator = torch.Generator().manual_seed(20260817)
    return (
        torch.rand(BATCH, 3, generator=generator),
        torch.rand(BATCH, 2, generator=generator),
    )


def _planner_fixture():
    generator = torch.Generator().manual_seed(11)
    labels = torch.randint(0, CLASSES, (BATCH, FRAMES), generator=generator)
    music = torch.rand(BATCH, FRAMES, 4, generator=generator)
    padding = torch.zeros(BATCH, FRAMES, dtype=torch.bool)
    return labels, music, padding


def _planner():
    torch.manual_seed(3)
    return UniformD3PM(
        AtomicPlannerTransformer(
            num_atomic_classes=CLASSES, music_dim=4, latent_dim=8,
            num_layers=1, num_heads=1, ff_size=8, dropout=0.0, max_seq_len=FRAMES,
        ),
        num_steps=4,
    )


def _init(rank, directory):
    torch.distributed.init_process_group(
        backend="gloo",
        init_method="file://{}".format(os.path.join(directory, "rendezvous")),
        rank=rank,
        world_size=WORLD_SIZE,
    )


def _quadratic_worker(rank, directory):
    _init(rank, directory)
    try:
        torch.manual_seed(0)
        model = _Quadratic()
        wrapped = DistributedDataParallel(model, broadcast_buffers=False)
        inputs, targets = _fixture()
        shard = slice(rank * (BATCH // WORLD_SIZE), (rank + 1) * (BATCH // WORLD_SIZE))
        wrapped(inputs[shard], targets[shard]).backward()
        torch.save(model.linear.weight.grad.detach().clone(),
                   os.path.join(directory, "quadratic_{}.pt".format(rank)))
    finally:
        torch.distributed.destroy_process_group()


def _planner_worker(rank, directory, mode):
    _init(rank, directory)
    try:
        model = _planner()
        wrapped = DistributedDataParallel(model, broadcast_buffers=False)
        labels, music, padding = _planner_fixture()
        shard = slice(rank * (BATCH // WORLD_SIZE), (rank + 1) * (BATCH // WORLD_SIZE))
        # Same RNG on both ranks, different data: any difference that survives
        # into the gradients therefore comes from the shard, which is exactly
        # what the reduction has to erase.
        torch.manual_seed(7)
        if mode == "wrapped":
            output = wrapped(labels[shard], music[shard], padding[shard])
        else:
            output = wrapped.module.training_step(labels[shard], music[shard], padding[shard])
        output.loss.backward()
        grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        torch.save(grads, os.path.join(directory, "planner_{}_{}.pt".format(mode, rank)))
    finally:
        torch.distributed.destroy_process_group()


def _run(worker, directory, *extra):
    # ``spawn``, although ``launch_distributed`` uses ``fork``.  The launcher
    # forks to share a 104 GiB prototype library; this test has nothing to share,
    # and forking here would make it depend on what ran before it in the same
    # pytest process -- a sibling test that has completed one backward pass
    # leaves autograd's engine unusable in every child.  Spawn is immune, and the
    # property under test is DDP's, not the start method's.
    context = torch.multiprocessing.start_processes(
        worker, args=(directory, *extra), nprocs=WORLD_SIZE, start_method="spawn", join=False
    )
    if not context.join(timeout=300):
        context.join()


class DistributedGradientTests(unittest.TestCase):
    def test_global_batch_split_across_ranks_reproduces_the_whole_batch_gradient(self):
        # The oracle is written out by hand rather than taken from a second
        # backward pass.  ``train_atomic.launch_distributed`` forks its ranks, and
        # a process that has already run one backward hands every forked child an
        # unusable autograd engine -- "Unable to handle autograd's threading in
        # combination with fork-based multiprocessing".  Keeping this test free of
        # parent-side backwards keeps the same discipline visible where the
        # launcher's constraint is easiest to forget.
        inputs, targets = _fixture()
        torch.manual_seed(0)
        reference = _Quadratic()
        with torch.no_grad():
            residual = reference.linear(inputs) - targets
            expected = (2.0 / residual.numel()) * (residual.T @ inputs)

        with tempfile.TemporaryDirectory() as directory:
            _run(_quadratic_worker, directory)
            grads = [torch.load(os.path.join(directory, "quadratic_{}.pt".format(rank)),
                                weights_only=True) for rank in range(WORLD_SIZE)]

        self.assertTrue(torch.allclose(grads[0], grads[1], atol=1e-7),
                        "ranks disagree, so nothing was reduced")
        self.assertTrue(
            torch.allclose(grads[0], expected, atol=1e-6),
            "the reduced gradient is not the whole-batch gradient: {} vs {}".format(grads[0], expected),
        )

    def test_wrapping_the_diffusion_module_is_what_synchronises_its_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            _run(_planner_worker, directory, "wrapped")
            wrapped = [torch.load(os.path.join(directory, "planner_wrapped_{}.pt".format(rank)),
                                  weights_only=True) for rank in range(WORLD_SIZE)]
            _run(_planner_worker, directory, "bypassed")
            bypassed = [torch.load(os.path.join(directory, "planner_bypassed_{}.pt".format(rank)),
                                   weights_only=True) for rank in range(WORLD_SIZE)]

        self.assertTrue(wrapped[0], "no gradients were produced at all")
        for name, gradient in wrapped[0].items():
            self.assertTrue(
                torch.allclose(gradient, wrapped[1][name], atol=1e-6),
                "rank gradients differ for {} after DDP forward".format(name),
            )
        # The control: the same fixture with the reduction skipped must leave the
        # ranks disagreeing.  If it does not, the check above proves nothing --
        # the shards would have produced identical gradients regardless.
        divergent = [
            name for name, gradient in bypassed[0].items()
            if not torch.allclose(gradient, bypassed[1][name], atol=1e-6)
        ]
        self.assertTrue(
            divergent,
            "bypassing DDP left every gradient identical, so this fixture cannot "
            "detect an unsynchronised run and the test above is vacuous",
        )


if __name__ == "__main__":
    unittest.main()
