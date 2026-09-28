import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "torch_scatter_compat"))

from torch_scatter import (  # noqa: E402
    broadcast,
    scatter_max,
    scatter_softmax,
    scatter_sum,
)


def reference_scatter_sum(src, index, dim, dim_size):
    """An explicit per-group loop; slow, obviously correct, and the pin."""
    size = list(src.size())
    size[dim] = dim_size
    out = torch.zeros(size, dtype=src.dtype)
    src_moved = src.movedim(dim, 0)
    out_moved = out.movedim(dim, 0)
    for position in range(src_moved.size(0)):
        out_moved[int(index[position])] += src_moved[position]
    return out_moved.movedim(0, dim)


def reference_scatter_softmax(src, index, dim, dim_size):
    src_moved = src.movedim(dim, 0)
    out = torch.zeros_like(src_moved)
    for group in range(dim_size):
        rows = [i for i in range(src_moved.size(0)) if int(index[i]) == group]
        if not rows:
            continue
        block = torch.stack([src_moved[i] for i in rows])
        weights = torch.softmax(block, dim=0)
        for offset, row in enumerate(rows):
            out[row] = weights[offset]
    return out.movedim(0, dim)


class BroadcastTests(unittest.TestCase):
    def test_expands_a_flat_index_to_the_source_shape_along_dim(self):
        src = torch.zeros(2, 5, 3)
        index = torch.tensor([0, 1, 1, 0, 2])
        expanded = broadcast(index, src, 1)
        self.assertEqual(expanded.shape, src.shape)
        # Every row of the expanded index repeats the same group id.
        self.assertTrue(torch.equal(expanded[0, :, 0], index))
        self.assertTrue(torch.equal(expanded[1, :, 2], index))

    def test_negative_dim_counts_from_the_end(self):
        src = torch.zeros(4, 6)
        index = torch.tensor([0, 0, 1, 1, 2, 2])
        self.assertTrue(torch.equal(broadcast(index, src, -1), broadcast(index, src, 1)))


class ScatterSumTests(unittest.TestCase):
    def test_matches_an_explicit_per_group_loop(self):
        torch.manual_seed(0)
        src = torch.randn(2, 7, 3)
        index = torch.tensor([0, 2, 1, 2, 0, 0, 3])
        got = scatter_sum(src, index, dim=1, dim_size=4)
        want = reference_scatter_sum(src, index, dim=1, dim_size=4)
        self.assertEqual(got.shape, want.shape)
        torch.testing.assert_close(got, want)

    def test_infers_dim_size_from_the_largest_index(self):
        src = torch.ones(1, 4)
        index = torch.tensor([0, 0, 3, 1])
        self.assertEqual(scatter_sum(src, index, dim=1).shape, (1, 4))

    def test_empty_index_yields_an_empty_axis(self):
        src = torch.ones(2, 0, 3)
        index = torch.zeros(0, dtype=torch.long)
        self.assertEqual(scatter_sum(src, index, dim=1).shape, (2, 0, 3))

    def test_groups_with_no_source_elements_stay_zero(self):
        src = torch.ones(1, 2)
        index = torch.tensor([0, 0])
        out = scatter_sum(src, index, dim=1, dim_size=3)
        torch.testing.assert_close(out, torch.tensor([[2.0, 0.0, 0.0]]))

    def test_writes_into_a_provided_output(self):
        out = torch.full((1, 3), 5.0)
        scatter_sum(torch.ones(1, 2), torch.tensor([0, 2]), dim=1, out=out)
        torch.testing.assert_close(out, torch.tensor([[6.0, 5.0, 6.0]]))


class ScatterSoftmaxTests(unittest.TestCase):
    def test_matches_a_per_group_torch_softmax(self):
        torch.manual_seed(1)
        src = torch.randn(2, 8, 3)
        index = torch.tensor([0, 0, 1, 2, 2, 2, 1, 0])
        got = scatter_softmax(src, index, dim=1, dim_size=3)
        want = reference_scatter_softmax(src, index, dim=1, dim_size=3)
        torch.testing.assert_close(got, want)

    def test_each_group_sums_to_one(self):
        torch.manual_seed(2)
        src = torch.randn(1, 6, 2)
        index = torch.tensor([0, 1, 0, 1, 0, 1])
        weights = scatter_softmax(src, index, dim=1, dim_size=2)
        totals = scatter_sum(weights, index, dim=1, dim_size=2)
        torch.testing.assert_close(totals, torch.ones_like(totals))

    def test_large_magnitudes_do_not_overflow(self):
        # Without the group-max shift this is inf/inf.
        src = torch.tensor([[[100.0], [101.0], [-100.0]]])
        index = torch.tensor([0, 0, 1])
        weights = scatter_softmax(src, index, dim=1, dim_size=2)
        self.assertTrue(bool(torch.isfinite(weights).all()))
        torch.testing.assert_close(
            weights[0, :2, 0], torch.softmax(torch.tensor([100.0, 101.0]), dim=0)
        )

    def test_refuses_integer_input(self):
        with self.assertRaises(ValueError):
            scatter_softmax(torch.ones(1, 3, dtype=torch.long), torch.tensor([0, 0, 1]), dim=1)


class UnshimmedTests(unittest.TestCase):
    def test_scatter_max_refuses_rather_than_guessing(self):
        with self.assertRaises(NotImplementedError):
            scatter_max(torch.ones(1, 3), torch.tensor([0, 0, 1]), dim=1)


if __name__ == "__main__":
    unittest.main()
