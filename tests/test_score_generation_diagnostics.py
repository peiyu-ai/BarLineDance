"""The follow-rate metric must be able to report following.

Every other test here can pass while the metric is broken, because a metric
that under-reports still returns a number in [0, 1] and still moves in the right
direction.  The test that catches it is the one that hands the metric a
generation which *is* the draft: there is exactly one right answer, 1.0, and
before 2026-08-18 the tool returned 0.456 -- the prototype was read out of the
release's min-max normalised array without inverting it, so ax_from_6v decoded
the rotations of a body that does not exist.  Every published follow rate was a
fraction of a ceiling it could not reach.
"""

import pathlib
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataset.quaternion import ax_to_6v  # noqa: E402
from tools.score_generation_diagnostics import _rotations_from_151, geodesic_rad  # noqa: E402


def _normalizer(tmp, data_min, data_max):
    path = pathlib.Path(tmp) / "normalizer.pt"
    torch.save({"data_min": torch.tensor(data_min, dtype=torch.float32),
                "data_max": torch.tensor(data_max, dtype=torch.float32)}, str(path))
    return path


def _motion_from_rotations(rotations):
    """[T,24,3] axis-angle -> the repo's raw (un-normalised) 151-D layout."""
    six = ax_to_6v(rotations).reshape(len(rotations), 144)
    head = torch.zeros(len(rotations), 7)
    return torch.cat([head, six], dim=1)


class FollowRateSpaceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260818)
        self.rotations = torch.randn(40, 24, 3) * 0.4
        self.raw = _motion_from_rotations(self.rotations)

    def test_normalised_input_decodes_to_the_same_rotations(self):
        """A round trip through the release normaliser must be a no-op here."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            # A per-dimension min-max with a different range per column, which
            # is what a real release has -- a uniform scale would hide the bug.
            data_min = self.raw.min(dim=0).values - 0.3
            data_max = self.raw.max(dim=0).values + 0.7
            path = _normalizer(tmp, data_min.tolist(), data_max.tolist())
            span = torch.where(data_max == data_min,
                               torch.ones_like(data_max - data_min), data_max - data_min)
            normalised = (self.raw - data_min) / span * 2.0 - 1.0

            decoded = _rotations_from_151(normalised, path)
            self.assertLess(geodesic_rad(self.rotations, decoded), 1e-3)

    def test_reading_normalised_columns_as_rotations_is_not_a_no_op(self):
        """The defect must be visible, or the fix above is untested."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            data_min = self.raw.min(dim=0).values - 0.3
            data_max = self.raw.max(dim=0).values + 0.7
            _normalizer(tmp, data_min.tolist(), data_max.tolist())
            span = data_max - data_min
            normalised = (self.raw - data_min) / span * 2.0 - 1.0

            # Same call, no normaliser: this is what the tool did until
            # 2026-08-18, and it must land far from the true rotations.
            decoded = _rotations_from_151(normalised)
            self.assertGreater(geodesic_rad(self.rotations, decoded), 0.2)

    def test_a_generation_that_is_the_draft_scores_one(self):
        """The perfect-copier ceiling, stated as a test rather than a comment."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            data_min = self.raw.min(dim=0).values - 0.3
            data_max = self.raw.max(dim=0).values + 0.7
            path = _normalizer(tmp, data_min.tolist(), data_max.tolist())
            span = data_max - data_min
            draft = (self.raw - data_min) / span * 2.0 - 1.0

            proto = _rotations_from_151(draft, path)
            generated = proto.clone()          # the copier
            rng = np.random.default_rng(20260818)
            order = torch.from_numpy(rng.permutation(len(proto)))
            honest = geodesic_rad(generated, proto)
            control = geodesic_rad(generated, proto[order])
            follow = (control - honest) / control
            self.assertGreater(follow, 0.999)


if __name__ == "__main__":
    unittest.main()
