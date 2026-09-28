#!/usr/bin/env python3
"""Which clips in an arm never went through retrieval, so no table can pool them.

WHAT THE FIELD IS.  Every generated clip's pickle under ``runs/<arm>/`` carries
``prototype_retrieval.safe_draft_condition_fraction``: the share of the plan's
ATOMIC (non-transition) frames for which ``_source_safe_draft``
(``infer_atomic.py``) found a prototype from a DIFFERENT recording.  1.0 means
the whole named plan was covered by source-safe retrieval; 0.0 means none of it
was; ``None`` means the plan named no atomic frame at all, so the fraction is
0/0 and ``infer_atomic._safe_draft_condition_fraction`` deliberately refuses to
call that 1.0 (its docstring records the cost of the old return value).

WHY A CLIP READS 0.0.  ``_source_safe_draft`` fails CLOSED: with no
``query_retrieval_group_id`` it cannot prove a prototype is not the query's own
recording, so it returns an ALL-ZERO draft with an ALL-ZERO mask rather than
risk retrieving the answer.  The policy is right.  What that clip's row then
measures is the completion model running on MUSIC ALONE -- a different
experiment from the arm it is filed under.

WHY THIS MODULE EXISTS.  Measured 2026-09-03/04 on ``runs/opt_guidance_g15``:
2 of the 20 eval clips (``wild_v5:7188505181892381984:clip000`` and
``wild_v5:7610414564962183545:clip000``) are in no split of ``windows.jsonl``,
have no retrieval group, and record 0.0.  The pipeline reported it in every one
of their pickles and NO scoring tool read the field, so 10% of every published
T-line table was an unconditioned control pooled in as if it were M5 output.
Excluding them moved one reading (the draft's stillness share) 0.351 -> 0.390.

THE ABSENT-FIELD RULE, and why it raises instead of passing.  CLAUDE.md 2 is
about gates that cannot fire.  If this module treated a missing field as "fine",
then an arm written by an older or a different code path -- exactly the arm most
likely to be wrong -- would sail through the one check built to catch it, and
the check would still print a reassuring header.  So a pickle that exists but
carries no ``prototype_retrieval.safe_draft_condition_fraction`` raises
``ArmSourcingError``.  Ground-truth directories (``full_pose`` only) are NOT
arms and must never be passed here; the callers apply the arms' verdict to
ground truth by scoring it on the same surviving clip list.

WHY THE EXCLUSION IS THE UNION OVER ARMS.  A table compares arms to each other.
Dropping a clip from one arm and keeping it in another compares different clip
sets, which is the mistake ``score_arm_table``'s own docstring records in
another form (whole-clip vs chunked ``lag0``).  So ``select_clips`` excludes a
clip from EVERY arm and from ground truth as soon as ONE arm reports it
unsourced, and names it.
"""

import pathlib
import pickle

FIELD = "safe_draft_condition_fraction"
SECTION = "prototype_retrieval"

# Fail-closed only.  A clip that got SOME source-safe coverage is still M5
# output; a clip that got none is not.  Raise it with --sourcing-threshold when
# a table needs "fully sourced or nothing".
DEFAULT_THRESHOLD = 0.0


class ArmSourcingError(RuntimeError):
    """An arm directory could not be asked the question, so nothing is claimed."""


def _read_fraction(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or SECTION not in payload:
        raise ArmSourcingError(
            "{}: no {!r} section.  A scoring tool cannot tell whether this "
            "clip's draft came from retrieval, so it must not be pooled into an "
            "arm's number.  If this is a ground-truth directory, pass it as "
            "--ground-truth-dir, not as --arm.".format(path, SECTION))
    section = payload[SECTION]
    if not isinstance(section, dict) or FIELD not in section:
        raise ArmSourcingError(
            "{}: {!r} carries no {!r}.  A missing gate must not read as a "
            "passing gate; regenerate this arm with the current "
            "infer_atomic.py.".format(path, SECTION, FIELD))
    value = section[FIELD]
    if value is None:
        return None
    return float(value)


class ArmSourcing(object):
    """Per-clip sourcing verdict for one arm directory.

    ``fractions``  clip -> float, or ``None`` for "the plan named no atomic
                   frame", which is NOT the same as "everything was sourced".
    ``unsourced``  clips at or below the threshold.
    ``undefined``  clips whose fraction is ``None``.
    ``missing``    clips with no pickle in this directory at all -- reported,
                   never silently equated with either of the above.
    """

    def __init__(self, name, directory, fractions, missing, threshold):
        self.name = name
        self.directory = str(directory)
        self.fractions = fractions
        self.missing = sorted(missing)
        self.threshold = float(threshold)
        self.unsourced = sorted(clip for clip, value in fractions.items()
                                if value is not None and value <= threshold)
        self.undefined = sorted(clip for clip, value in fractions.items()
                                if value is None)

    @property
    def excluded(self):
        """Both refusals, together: neither is M5 output on a named plan."""
        return sorted(set(self.unsourced) | set(self.undefined))

    def header(self):
        return {
            "arm": self.name,
            "directory": self.directory,
            "threshold": self.threshold,
            "clips_read": len(self.fractions),
            "unsourced_count": len(self.unsourced),
            "unsourced_clips": self.unsourced,
            "undefined_count": len(self.undefined),
            "undefined_clips": self.undefined,
            "missing_pickles": self.missing,
            "fractions": {clip: self.fractions[clip] for clip in sorted(self.fractions)},
        }

    def line(self):
        return ("sourcing[{}] {} clips read, {} unsourced (<= {:g}), {} undefined"
                .format(self.name, len(self.fractions), len(self.unsourced),
                        self.threshold, len(self.undefined)))


def load_arm_sourcing(directory, clips=None, name=None, threshold=DEFAULT_THRESHOLD):
    """Read ``safe_draft_condition_fraction`` for every clip of one arm.

    ``clips`` restricts to a clip list (the eval set); ``None`` reads every
    pickle in the directory.  Raises ``ArmSourcingError`` when the directory is
    absent, when it holds no readable clip at all (a gate that reads nothing
    cannot fire), or when a pickle carries no fraction.
    """
    directory = pathlib.Path(directory)
    name = name or directory.name
    if not directory.is_dir():
        raise ArmSourcingError("{}: not a directory, so no arm was read for "
                               "{!r}.".format(directory, name))
    if clips is None:
        wanted = sorted(p.stem for p in directory.glob("*.pkl"))
    else:
        wanted = list(clips)
    fractions, missing = {}, []
    for clip in wanted:
        path = directory / (clip + ".pkl")
        if not path.is_file():
            missing.append(clip)
            continue
        fractions[clip] = _read_fraction(path)
    if not fractions:
        raise ArmSourcingError(
            "{}: none of the {} requested clips has a pickle here, so the "
            "sourcing gate read nothing.  A gate that reads nothing is not a "
            "gate that passed.".format(directory, len(wanted)))
    return ArmSourcing(name, directory, fractions, missing, threshold)


def select_clips(clips, arms, threshold=DEFAULT_THRESHOLD, include_unsourced=False):
    """The clip list every arm AND ground truth is scored on, plus the header.

    ``arms`` is an iterable of ``(name, directory)``.  Returns
    ``(kept_clips, header)``; ``header`` always states the policy, the count and
    the names, so a filtered number in a JSON file cannot be mistaken for an
    unfiltered one -- and an UNfiltered run says so in the same field rather
    than omitting it.
    """
    clips = list(clips)
    per_arm = [load_arm_sourcing(directory, clips, name=name, threshold=threshold)
               for name, directory in arms]
    excluded = sorted({clip for arm in per_arm for clip in arm.excluded})
    kept = clips if include_unsourced else [c for c in clips if c not in excluded]
    header = {
        "sourcing_policy": ("INCLUDE_UNSOURCED_BACKWARD_COMPARISON"
                            if include_unsourced else "EXCLUDE_UNSOURCED"),
        "sourcing_field": "{}.{}".format(SECTION, FIELD),
        "sourcing_threshold": float(threshold),
        "clips_requested": len(clips),
        "clips_scored": len(kept),
        "unsourced_excluded_count": 0 if include_unsourced else len(excluded),
        "unsourced_excluded_clips": [] if include_unsourced else excluded,
        "unsourced_detected_count": len(excluded),
        "unsourced_detected_clips": excluded,
        "per_arm": [arm.header() for arm in per_arm],
    }
    return kept, header


def format_header(header):
    """The lines a scoring tool prints above its table."""
    lines = ["sourcing: {} field={} threshold<={:g}".format(
        header["sourcing_policy"], header["sourcing_field"],
        header["sourcing_threshold"])]
    detected = header["unsourced_detected_clips"]
    if not detected:
        lines.append("sourcing: 0 unsourced clips; {} of {} scored".format(
            header["clips_scored"], header["clips_requested"]))
    else:
        kept_in = header["sourcing_policy"] == "INCLUDE_UNSOURCED_BACKWARD_COMPARISON"
        lines.append("sourcing: {} unsourced clip(s) {}: {}".format(
            len(detected),
            "KEPT (--include-unsourced)" if kept_in else "EXCLUDED",
            ", ".join(detected)))
        lines.append("sourcing: {} of {} clips scored".format(
            header["clips_scored"], header["clips_requested"]))
    for arm in header["per_arm"]:
        if arm["missing_pickles"]:
            lines.append("sourcing[{}]: {} requested clip(s) have no pickle: {}".format(
                arm["arm"], len(arm["missing_pickles"]),
                ", ".join(arm["missing_pickles"])))
    return lines


def add_arguments(parser):
    """The two flags every wired-in scoring tool exposes, worded identically."""
    parser.add_argument("--include-unsourced", action="store_true",
                        help="pool clips whose draft was never sourced "
                             "(safe_draft_condition_fraction <= threshold) back "
                             "in, for comparison with numbers published before "
                             "this gate existed")
    parser.add_argument("--sourcing-threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="a clip is unsourced when its "
                             "safe_draft_condition_fraction is <= this "
                             "(default %(default)g, fail-closed only)")
    return parser
