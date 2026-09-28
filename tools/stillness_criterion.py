"""The ONE definition of "does the body ever land", and its refusal.

WHY THIS FILE EXISTS.  The sustained-stillness criterion had three separate
implementations (``measure_motion_dynamics.dynamics``,
``exp_guidance_stillness.held_frames``, ``exp_draftonly_stillness_windows.
window_contains_hold``) that agreed on the definition and DISAGREED on what to
do with an input the definition cannot describe.  On the two eval clips whose
draft is a motionless body (DANCE_QUALITY_DEFECTS.md 23.9: ``_source_safe_draft``
fails closed and hands back an all-zero draft) ``measure_motion_dynamics``
silently DROPPED them -- printing ``clips: 16`` where 20 were asked for -- while
``exp_guidance_stillness`` silently printed ``0.0``.  Both readings are wrong in
the same way and in opposite directions, which is the shape CLAUDE.md 2 names:
a gate that cannot fail, reading like a measurement.

WHAT DEGENERATES, ALGEBRAICALLY.  A held frame is one whose smoothed
root-relative joint speed is below ``HOLD_FRACTION`` x THAT clip's (or window's)
own median smoothed speed.  The threshold is the clip's own scale on purpose: an
absolute threshold would call a slow song one long hold.  But when the median is
zero the threshold is zero, and ``speed < 0`` is unsatisfiable for a
non-negative quantity -- so a body that is still in EVERY frame reads
``hold_share = 0.0``.  The reading is not small, it is INVERTED.  The same
collapse hits the ``p90/p10`` dynamic-range column from the other side: at
``p10 == 0`` the old code divided by ``max(p10, 1e-9)``, fabricating ratios of
~1e11 out of a clamp constant.

THE REFUSAL, AND WHERE ITS THRESHOLD COMES FROM.  A clip or window is
measurable iff its median smoothed speed exceeds ``MEASURABLE_MEDIAN_SPEED``,
and its dynamic range is measurable iff its raw p10 does.  The floor is NOT
fitted to the arms it judges; it is the float32 storage noise of the pkl the
motion is read from.  Joint coordinates are O(1 m) and stored float32, whose
spacing there is 1.19e-7 m; a one-ulp change per frame at 30 fps is 3.6e-6 m/s.
``MEASURABLE_MEDIAN_SPEED = 1e-4`` m/s is ~28x that -- 3 microns of travel per
frame -- so anything below it is indistinguishable from a constant pose written
to disk, and nothing a body does can hide under it.

MEASURED SEPARATION (2026-09-04, the 20 held-out T-line clips; whole-clip plus
50 x 150-frame windows per clip, over ground truth / retrieval draft /
draft+SG5/9/15 / g1.0 / g1.5 / shipping g2.0 / ORACLE-gt = 5100 readings):

  degenerate readings      600   ALL EXACTLY 0.000000 m/s
  measurable readings     4500   minimum 0.001482 m/s (a smoothed draft window)
                                 1st percentile 0.023393 m/s
  ground truth            1020   minimum 0.147957 m/s

Nothing in the corpus lands in ``(0, 1e-4)``, so the floor's exact placement
anywhere between the float32 rate (3.6e-6) and the smallest real reading
(1.5e-3) changes no verdict on this data.  The separation is bimodal because
the degeneracy is not "slow", it is "written as a constant".

TWO CANDIDATE TESTS WERE REJECTED, and their numbers are why (CLAUDE.md 2.1
rule 4 -- when two rulers disagree, say so rather than pick the nice one):

  * ``p90/p10 below a bound``.  NON-MONOTONE in the degeneracy.  A fully frozen
    draft clip reads 0.0 (0 / clamp), but a HALF-frozen one --
    ``wild_v5:7148005618245242151:clip000``, 348 of 664 smoothed frames exactly
    zero -- reads 4.5e11.  A "below a bound" test catches the first and misses
    the second, and the second is just as degenerate: its median is 0 too.
  * ``IQR below a floor``.  Same failure.  That clip's speed IQR is 0.2989 m/s,
    larger than fourteen of the twenty ground-truth clips', while its median is
    exactly 0.  IQR does not see the collapse of the threshold because the
    threshold is built from the MEDIAN, not from the spread.

The median floor is not a third guess: it is the exact quantity the criterion
divides by, so it is the quantity whose vanishing defines the degeneracy.

WHAT CALLERS MUST DO.  ``held_frames`` and ``dynamic_range`` RAISE
``DegenerateMotion`` instead of returning a number.  Every caller catches it and
records the input in a ``DropLog``, whose ``header()`` goes into the emitted
JSON as ``clips_measured`` / ``clips_dropped`` / ``dropped_clips``; a CLI whose
arm has nothing measurable left exits with the reason.  A dropped clip is never
a silent zero and never an invisible skip.
"""

import numpy as np

FPS = 30.0
SMOOTH_WIDTH = 9
HOLD_FRACTION = 0.25
# See the module docstring: float32 storage noise of the pose, not a fit.
MEASURABLE_MEDIAN_SPEED = 1e-4


class DegenerateMotion(ValueError):
    """The self-relative stillness criterion has no scale on this input.

    ``scale`` is the reading that vanished (median smoothed speed, or raw p10
    for the dynamic range) and ``statistic`` names which one, so a caller can
    put the reason in the JSON rather than only the fact of the drop.
    """

    def __init__(self, statistic, scale, *, frames=None, what="input"):
        self.statistic = statistic
        self.scale = float(scale)
        self.frames = frames
        self.what = what
        super().__init__(
            "{} is not measurable by the self-relative stillness criterion: "
            "{} = {:.3e} m/s, at or below the {:.0e} m/s float32 storage floor"
            "{}.  The criterion's threshold is a fraction of this quantity, so "
            "at zero the threshold is zero and a motionless body would read "
            "'no stillness'.  Refusing to emit a number."
            .format(what, statistic, self.scale, MEASURABLE_MEDIAN_SPEED,
                    "" if frames is None else " ({} frames)".format(frames)))


def joint_speed(joints):
    """Root-relative mean joint speed per frame, m/s."""
    joints = np.asarray(joints, float)
    relative = joints - joints[:, :1, :]
    return np.linalg.norm(np.diff(relative, axis=0), axis=2).mean(axis=1) * FPS


def low_pass(joints, width=SMOOTH_WIDTH):
    """Moving average over joint positions, separating a landing from a dip."""
    joints = np.asarray(joints, float)
    if width <= 1:
        return joints
    kernel = np.ones(width) / width
    flat = joints.reshape(len(joints), -1)
    padded = np.pad(flat, ((width, width), (0, 0)), mode="edge")
    out = np.stack([np.convolve(padded[:, i], kernel, mode="same")
                    for i in range(flat.shape[1])], axis=1)
    return out[width:-width].reshape(joints.shape)


def longest_run(mask):
    best = current = 0
    for value in np.asarray(mask, bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def speed_scale(speed, *, what="input"):
    """The median the criterion divides by, or raise if it has vanished."""
    speed = np.asarray(speed, float)
    if speed.size < 3:
        raise DegenerateMotion("frames", 0.0, frames=int(speed.size),
                               what=what)
    median = float(np.median(speed))
    if not (median > MEASURABLE_MEDIAN_SPEED):
        raise DegenerateMotion("median smoothed speed", median,
                               frames=int(speed.size), what=what)
    return median


def held_frames(speed, hold_fraction=HOLD_FRACTION, *, what="input"):
    """Boolean mask of held frames.  RAISES on a body with no scale."""
    speed = np.asarray(speed, float)
    return speed < hold_fraction * speed_scale(speed, what=what)


def dynamic_range(speed, *, what="input"):
    """p90/p10 of speed.  RAISES rather than divide by a clamp constant."""
    speed = np.asarray(speed, float)
    if speed.size < 3:
        raise DegenerateMotion("frames", 0.0, frames=int(speed.size), what=what)
    low = float(np.percentile(speed, 10))
    if not (low > MEASURABLE_MEDIAN_SPEED):
        raise DegenerateMotion("raw p10 speed", low, frames=int(speed.size),
                               what=what)
    return float(np.percentile(speed, 90) / low)


def is_measurable(speed):
    """True iff ``held_frames`` would return rather than raise."""
    try:
        speed_scale(speed)
    except DegenerateMotion:
        return False
    return True


class DropLog:
    """Accounting for inputs the criterion refused, for the JSON header.

    Exists because both failure modes this module replaces were INVISIBLE: one
    tool skipped the clip and printed a smaller ``clips:`` count with no names,
    the other printed 0.0.  Anything a report is built on has to say how many
    inputs it could not measure and which ones.
    """

    def __init__(self, kind="clips"):
        self.kind = kind
        self.measured = []
        self.dropped = {}

    def measure(self, name, function, *arguments, **keywords):
        """Run ``function``; on refusal record ``name`` and return None."""
        try:
            value = function(*arguments, **keywords)
        except DegenerateMotion as refusal:
            self.drop(name, refusal)
            return None
        self.measured.append(name)
        return value

    def drop(self, name, reason):
        if isinstance(reason, DegenerateMotion):
            reason = "{} = {:.3e} m/s <= {:.0e}".format(
                reason.statistic, reason.scale, MEASURABLE_MEDIAN_SPEED)
        self.dropped[name] = str(reason)

    def header(self):
        return {
            "{}_measured".format(self.kind): len(self.measured),
            "{}_dropped".format(self.kind): len(self.dropped),
            "dropped_{}".format(self.kind): dict(sorted(self.dropped.items())),
        }

    def lines(self):
        out = ["{} measured {}, dropped {}".format(
            self.kind, len(self.measured), len(self.dropped))]
        for name, reason in sorted(self.dropped.items()):
            out.append("  DROPPED {}: {}".format(name, reason))
        return out
