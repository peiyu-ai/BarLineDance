#!/usr/bin/env python3
"""One reader for "which clips must be derived again", shared by every stage.

Stage B and stage F both need the same list and would otherwise each parse it.
This repo has already paid for that shape once -- four parsers disagreeing about
what a song's identity was -- so the manifest has one reader and the stages ask
it a question rather than reading a file.

Three inputs are accepted, because all three exist and hand-copying between
them is where a measurement stops being one:

* a plain list, one clip stem per line;
* ``tools/audit_clip_freshness.py`` output, whose ``stale`` maps a stage name
  to the clips whose artifacts were built from bytes the clip no longer has;
* ``tools/refix_wild_fps_clips_census.py`` output, whose ``clips`` is every
  clip an upload set currently produces.

An audit that names no stale clip for the stage asked about returns an empty
list, and that is a real answer -- "nothing to redo" -- distinct from a file
that does not describe this stage at all, which raises.  A caller cannot tell
those apart from an empty list, and reading "the corpus is clean" off a
mis-typed path is the failure this repo keeps meeting.
"""

from __future__ import annotations

import json
import pathlib
from typing import List, Sequence

STAGES = ("3d", "s3d")


class ManifestError(RuntimeError):
    pass


def load(path, stage: str) -> List[str]:
    """Clip stems named for re-derivation at ``stage``."""
    if stage not in STAGES:
        raise ManifestError("unknown stage {!r}; known: {}".format(
            stage, ", ".join(STAGES)))
    path = pathlib.Path(path)
    text = path.read_text(encoding="utf-8")
    if not text.lstrip().startswith(("{", "[")):
        return [line.strip() for line in text.splitlines() if line.strip()]

    parsed = json.loads(text)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    stale = parsed.get("stale")
    if isinstance(stale, dict):
        if stage not in stale:
            raise ManifestError(
                "{}: a freshness audit that did not cover stage {!r} (it has: "
                "{}).  Re-run the audit with --stages including it rather than "
                "reading this as 'nothing to redo'".format(
                    path, stage, ", ".join(sorted(stale)) or "nothing"))
        return list(stale[stage])
    if "clips" in parsed:
        clips = parsed["clips"]
        if isinstance(clips, dict):
            return sorted(clips)
        # ``runs/wild_ingest_v1_worklist.json`` stores rows, not names:
        # ``{"clip": "<stem>", "bbx": true}``.  ``map(str, ...)`` over those
        # produces the repr of a dict -- a stem list every consumer accepts and
        # no clip matches, so a stage driven by it does nothing and reports the
        # count it was asked for.  A row without a ``clip`` key is a file this
        # reader does not understand, which is a different answer from "no
        # clips".
        if clips and isinstance(clips[0], dict):
            missing = [row for row in clips if "clip" not in row]
            if missing:
                raise ManifestError(
                    "{}: {} of {} clip rows have no 'clip' key; first: {}".format(
                        path, len(missing), len(clips), missing[0]))
            return sorted(str(row["clip"]) for row in clips)
        return sorted(map(str, clips))
    raise ManifestError(
        "{}: no stem list found (looked for stale.{}, clips)".format(path, stage))


def partition(named: Sequence[str], known: Sequence[str]):
    """Split a redo list into stems the corpus knows and stems it does not.

    Kept separate rather than filtered, because the two need opposite handling
    and both are silent failures if merged.  A named stem the corpus cannot
    serve means the caller believes it asked for work that will not happen; a
    named stem missing from a *worklist* is usually a clip that did not exist
    when the worklist was frozen, which is exactly what a re-cut produces.
    """
    known_set = set(known)
    inside = [stem for stem in named if stem in known_set]
    outside = [stem for stem in named if stem not in known_set]
    return inside, outside
