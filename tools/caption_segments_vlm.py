#!/usr/bin/env python3
"""Caption each atomic segment with a video-language model (paper M3, step 1).

The paper captions every segment with Gemini-2.5-Pro and then has a summarizing
LLM group mutually similar captions into sub-prototypes.  Gemini is not
reachable from this network, so Qwen3-VL runs in its place; the substitution is
recorded in the output header and travels into every label row downstream, the
same way S3D standing in for I3D is recorded.

What matters here is *not* caption eloquence.  These captions exist to be
clustered, so two instances of the same movement must come back in near
identical words or they will land in different sub-prototypes.  Free prose does
the opposite: it varies the wording precisely where the model has latitude.  So
the prompt asks for a small JSON object over a closed vocabulary -- the fields
are the axes dance movements actually differ along -- and decoding is greedy.
The free-text ``summary`` is kept for humans to read, but the structured fields
are what the embedding sees.

Frames come from the source video through the sequence's ``frame_ids.npy``,
which is the only honest mapping: motion frame i was extracted from source
frame ``frame_ids[i]``, and clips do not all start at zero.

``--posescript`` adds the paper's other input to this step: PoseScript
descriptions of the keyframes at motion beats, "provided as auxiliary cues to
the VLM descriptor".  They come from the 3D estimate of the same frames, so the
prompt presents them as a hint and tells the model to believe the frames where
the two disagree -- a wrong pose estimate should not be able to overwrite what
is visible.

Usage:
    python3 tools/caption_segments_vlm.py \
        --labels data/wild3d/wild_tmr_labels_v1 \
        --bundle data/wild3d/wild_performance_v1 \
        --video-dir data/wild_videos_converted \
        --model third_party/QwenVL/Qwen3-VL-30B-A3B-Instruct \
        --output runs/wild_captions_v1/captions.jsonl \
        --shard 0 --num-shards 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import time
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.cluster_atomics_tmr import build_row_index, resolve_row  # noqa: E402
from tools.recluster_atomics_ingroup import segments_of  # noqa: E402

SCHEMA_VERSION = "atomicdance-segment-captions-v1"
SCHEMA_VERSION_V2 = "atomicdance-segment-captions-v2"
SCHEMA_VERSION_V2DRAFT = "atomicdance-segment-captions-v2draft"
PRODUCER_VERSION = "qwen-vl-segment-caption-v1"

# A closed vocabulary is the whole point: the sub-clustering downstream can only
# separate movements along axes the captions actually encode, and it can only
# group two samples that describe the same movement the same way.  These six
# axes are the ones dance annotation schemes converge on.
FIELDS = ("body_action", "arms", "legs", "level", "travel", "dynamics")
VOCABULARY = {
    "body_action": ["step", "turn", "jump", "slide", "wave", "bounce", "pose",
                    "drop", "kick", "spin", "sway", "lean", "roll", "shuffle"],
    "arms": ["raised", "extended", "crossed", "down", "swinging", "circling",
             "framing_face", "pushing", "waving", "still"],
    "legs": ["together", "apart", "crossing", "lifted", "bent", "stepping",
             "kicking", "stationary", "crouched"],
    "level": ["high", "middle", "low", "changing"],
    "travel": ["in_place", "forward", "backward", "sideways", "rotating"],
    "dynamics": ["sharp", "smooth", "bouncy", "sustained", "explosive", "slow"],
}

# ---------------------------------------------------------------- schema v2
# Measured against the paper's own specification of what a caption must carry:
# "movement dynamics ... encompassing the path, rhythm, intensity, and fluidity
# of motion".  Schema v1 covers path (``travel``) and puts intensity and
# fluidity together in one single-valued ``dynamics`` axis, which makes them
# mutually exclusive -- a movement cannot be recorded as both explosive and
# smooth -- and has no field for rhythm at all.  Measured on clean5b5, v1
# captions mention a rhythmic or temporal property in 0 of 10 sampled segments
# while free text from the same model on the same frames mentions one in 9 of
# 10, so the omission is the vocabulary's and not the model's.
#
# **"Rhythm" here is the movement's own temporal pattern, not its relation to
# the music.**  This captioner is given frames and no audio, so an off-beat
# accent is not observable to it; ``syncopated`` was drafted and dropped for
# that reason.  Every value below is a property of the motion alone, which is
# also what makes them testable against the 3D by
# ``tools/probe_caption_grounding.py``.
# ``rhythm`` was drafted, measured and **rejected**, and the draft is kept so
# the negative result stays reproducible.  Measured 2026-08-22 on 255 segments,
# both from six stills and from thirty-two frames of real video:
#
#     rhythm=held          0.806 / 0.817   (grounded -- but "held" is "barely
#                                            moving", which is not a rhythm)
#     rhythm=steady        0.534 / 0.551   (chance)
#     rhythm=stop_and_go   0.514 / 0.537   (chance)
#     rhythm=pulsing       0.481 / 0.468   (chance)
#
# and ``pulsing`` was then checked against four separate rulers -- pelvis
# vertical reversals, whole-body vertical reversals, speed-peak count, and the
# autocorrelation of the speed signal -- which read 0.481, 0.480, 0.420 and
# **0.332**.  The sharpest of the four is reliably *backwards*: segments the
# model calls pulsing are less periodic than the rest.  So this is not a ruler
# that was too narrow, and it is not fixed by showing the model more frames.
#
# The paper lists rhythm as a component of movement dynamics.  Leaving it out
# is therefore a stated gap in this reproduction -- but a measured one, which
# is worth more than an axis the vocabulary carries and cannot fill.
FIELDS_V2DRAFT = ("body_action", "arms", "legs", "level", "travel",
                  "intensity", "fluidity", "rhythm")
FIELDS_V2 = ("body_action", "arms", "legs", "level", "travel",
             "intensity", "fluidity")
VOCABULARY_V2 = dict(
    body_action=list(VOCABULARY["body_action"]),
    arms=list(VOCABULARY["arms"]),
    legs=list(VOCABULARY["legs"]),
    level=list(VOCABULARY["level"]),
    travel=list(VOCABULARY["travel"]),
    # How much force the movement carries.
    intensity=["explosive", "strong", "moderate", "gentle"],
    # How continuous it is.  ``sharp`` and ``staccato`` are separated by how
    # many breaks there are, not by how hard they are -- that is what
    # ``intensity`` is for, and keeping the two apart is the entire point of
    # splitting v1's ``dynamics``.
    fluidity=["flowing", "sustained", "sharp", "staccato"],
    # The temporal shape of the movement itself.  ``bouncy`` lived in v1's
    # ``dynamics`` and is a rhythm, not a quality of force or of flow; it is
    # ``pulsing`` here.
    rhythm=["steady", "pulsing", "accelerating", "decelerating",
            "stop_and_go", "held"],
)

SENTENCE_V1 = ("a person performs a {body_action} with arms {arms} and legs "
               "{legs}, at {level} level, moving {travel}, {dynamics}")
SENTENCE_V2DRAFT = ("a person performs a {body_action} with arms {arms} and legs "
                    "{legs}, at {level} level, moving {travel}, {intensity} and "
                    "{fluidity}, in a {rhythm} rhythm")
SENTENCE_V2 = ("a person performs a {body_action} with arms {arms} and legs "
               "{legs}, at {level} level, moving {travel}, {intensity} and "
               "{fluidity}")

SCHEMAS = {
    "v1": {"fields": FIELDS, "vocabulary": VOCABULARY,
           "sentence": SENTENCE_V1, "version": SCHEMA_VERSION},
    "v2": {"fields": FIELDS_V2, "vocabulary": VOCABULARY_V2,
           "sentence": SENTENCE_V2, "version": SCHEMA_VERSION_V2},
    "v2draft": {"fields": FIELDS_V2DRAFT, "vocabulary": VOCABULARY_V2,
                "sentence": SENTENCE_V2DRAFT,
                "version": SCHEMA_VERSION_V2DRAFT},
}
# v1 stays the default so every existing importer, artifact and test keeps the
# behaviour it was written against; the schema in force is recorded on every
# caption row rather than inferred from when the run happened.
_schema = SCHEMAS["v1"]


def active_schema() -> Dict[str, object]:
    return _schema


def use_schema(name: str) -> Dict[str, object]:
    global _schema
    if name not in SCHEMAS:
        raise CaptionError("unknown schema {!r}; have {}".format(
            name, sorted(SCHEMAS)))
    _schema = SCHEMAS[name]
    return _schema


PROMPT = (
    "These frames are consecutive moments of one short dance movement.\n"
    "Describe the movement by filling this JSON exactly. Choose every value "
    "from the allowed list; do not invent values, do not add fields.\n\n"
    "{schema}\n\n"
    'Then add "summary": one short phrase naming the move.\n'
    "Reply with the JSON object only."
)

# The paper's auxiliary cue: "we use PoseScript to describe selected keyframes
# identified as motion beats... These PoseScript annotations are provided as
# auxiliary cues to the VLM descriptor".  It is stated as a hint, not as ground
# truth, because it comes from the 3D estimate -- which is itself derived from
# these same frames and can be wrong about them.  Saying so in the prompt is
# what keeps a bad pose estimate from overriding what the model can see.
CUE = ("\n\nA 3D pose estimate of the same moments reads: {cue}.\n"
       "Treat it as a hint for what the frames leave ambiguous; where it "
       "disagrees with the frames, believe the frames.")


class CaptionError(RuntimeError):
    pass


def dtype_kwarg(version: str) -> str:
    """Name this ``transformers`` gives the weight-dtype argument.

    ``torch_dtype`` was renamed to ``dtype`` in 4.56.  The old name still works
    there, but the new one does *not* work before it: 4.51 forwards unrecognised
    keywords into the model constructor, so ``dtype`` fails only after the
    weights have been located, with a TypeError naming the model class rather
    than the argument.  Two Qwen generations are in play here -- Qwen2.5-VL
    loads under the older transformers and Qwen3-VL needs the newer one -- so
    the captioner has to speak both.
    """
    parts = []
    for chunk in version.split(".")[:2]:
        match = re.match(r"\d+", chunk)
        parts.append(int(match.group(0)) if match else 0)
    return "dtype" if tuple(parts) >= (4, 56) else "torch_dtype"


def schema_text() -> str:
    """The allowed-values block for whichever schema is in force."""
    lines = ["{"]
    for field in _schema["fields"]:
        lines.append('  "{}": one of {},'.format(
            field, _schema["vocabulary"][field]))
    lines.append('  "summary": "<short phrase>"')
    lines.append("}")
    return "\n".join(lines)


def model_name(model_dir: pathlib.Path) -> str:
    """A name that identifies the model, not the revision directory it sits in.

    The OSS mirror stores each model under a revision folder, so the documented
    path ends ``.../Qwen3-VL-30B-A3B-Instruct/main`` and the directory's own
    name is ``main``.  That name travels into every caption row and from there
    into every label row as the thing standing in for Gemini-2.5-Pro, so
    recording it as "main" would leave the substitution unattributable.
    """
    resolved = model_dir.resolve()
    if resolved.name in ("main", "snapshots", "model") and resolved.parent.name:
        return "{}".format(resolved.parent.name)
    return resolved.name


def shard_of(key: Tuple[str, int, int], num_shards: int) -> int:
    """Which worker owns this segment.

    Deliberately *not* ``hash()``: Python salts string hashing per process, so
    eight workers would each compute a different partition of the same corpus
    -- some segments captioned several times with no way to tell which caption
    is which, others never captioned at all, and the split changing on every
    resume.  A digest is stable across processes and runs.
    """
    digest = hashlib.sha256("{}|{}|{}".format(*key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def sample_frame_positions(start: int, end: int, count: int) -> List[int]:
    """Evenly spaced motion-frame indices covering [start, end).

    Endpoints are included because a movement's identity often lives in where
    it starts and where it ends -- a step out and a step back share every
    interior frame.
    """
    length = end - start
    if length <= count:
        return list(range(start, end))
    if count == 1:
        return [start]
    return [start + int(round(i * (length - 1) / (count - 1))) for i in range(count)]


def parse_caption(text: str) -> Optional[Dict[str, str]]:
    """Pull the JSON object out of a reply, keeping only known field/value pairs.

    A value outside the vocabulary is dropped rather than kept: an off-schema
    value is a token the clustering has never seen anywhere else, so it would
    act as a unique fingerprint and pull its segment away from its neighbours.
    Dropping it degrades that field to "unspecified", which is honest.
    """
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    parsed: Dict[str, str] = {}
    for field in _schema["fields"]:
        value = raw.get(field)
        if isinstance(value, list) and value:
            value = value[0]
        if (isinstance(value, str)
                and value.strip().lower() in _schema["vocabulary"][field]):
            parsed[field] = value.strip().lower()
        else:
            parsed[field] = "unspecified"
    summary = raw.get("summary")
    parsed["summary"] = summary.strip() if isinstance(summary, str) else ""
    return parsed


def caption_sentence(parsed: Dict[str, str]) -> str:
    """Render the structured fields as one deterministic sentence.

    Deterministic word order is what makes two identical field sets produce
    byte-identical text, which is what makes them embed to the same point.
    """
    return _schema["sentence"].format(
        **{field: parsed.get(field, "unspecified").replace("_", " ")
           for field in _schema["fields"]})


def posescript_cue(segment_joints: np.ndarray, *, max_beats: int,
                   describer: Optional[object] = None,
                   segment_axis_angle: Optional[np.ndarray] = None) -> str:
    """PoseScript-style sentences for this segment's motion-beat keyframes.

    Beats repeat their description often -- a bounce settles into the same shape
    twice -- and a repeated clause costs prompt tokens while adding nothing, so
    consecutive duplicates collapse.

    ``describer`` is PoseScript's released caption model when it is available
    (see ``tools/posescript_capgen.py``); without it the rule-based fallback in
    ``tools.motion_beats`` writes the sentence instead.  The two are recorded
    under different names in every caption row, because "PoseScript said so" and
    "our posecode rules said so" are different claims.  Note which one reads
    what: beats come from joint *speed* either way, but the released model is
    fed axis-angle, not joint coordinates.
    """
    from tools.motion_beats import canonical_pose, describe_pose, find_motion_beats

    beats = find_motion_beats(segment_joints, max_beats=max_beats)
    if describer is not None and segment_axis_angle is not None:
        texts = describer.describe_many([segment_axis_angle[beat] for beat in beats])
    else:
        texts = [describe_pose(canonical_pose(segment_joints[beat])) for beat in beats]

    phrases: List[str] = []
    for text in texts:
        if text and (not phrases or phrases[-1] != text):
            phrases.append(text)
    return "; then ".join(phrases)


def iter_segments(labels_dir: pathlib.Path, bundle: pathlib.Path, *,
                  min_frames: int, posescript: bool = False, max_beats: int = 4,
                  wanted: Optional[Callable[[Tuple[str, int, int]], bool]] = None,
                  person_boxes: Optional[pathlib.Path] = None,
                  describer: Optional[object] = None,
                  embedding_cache: Optional[pathlib.Path] = None
                  ) -> Iterator[Dict[str, object]]:
    """Walk every accepted segment, optionally with its PoseScript cue.

    ``wanted`` is the caller's shard filter, pushed down here rather than
    applied after the fact.  Forward kinematics for a whole recording costs
    ~120 ms, and with eight shards walking the same corpus, computing cues for
    segments this worker will discard means every worker pays the full-corpus
    FK bill on the process that is meant to be feeding a GPU.

    ``embedding_cache`` supplies the segment boundaries M2 clustered.  Without
    it the boundaries are runs of equal frame label, which merges two adjacent
    segments whenever M2 gave them the same prototype -- 32.2% of them on
    AIST++ aist_v1.  That is not only a lost caption: the caption rows are
    keyed ``(recording, start, end)`` and ``recluster_atomics_ingroup`` matches
    them by that key, so captioning runs while re-clustering segments produces
    a *mismatch*, and the coverage gate rejects the pair after the GPU time has
    already been spent.  Both stages must be told the same thing.
    """
    rows = {json.loads(line)["recording_id"]: json.loads(line)
            for line in (bundle / "sequences.jsonl").open(encoding="utf-8")}
    index = build_row_index(rows)
    spans: Optional[Dict[str, List[Tuple[int, int]]]] = None
    if embedding_cache is not None:
        cached = np.load(embedding_cache, allow_pickle=True)
        for key in ("recordings", "starts", "ends"):
            if key not in cached:
                raise SystemExit("error: {} has no '{}'; it is not a segment-level "
                                 "embedding cache".format(embedding_cache, key))
        spans = {}
        for name, start, end in zip(cached["recordings"], cached["starts"], cached["ends"]):
            spans.setdefault(str(name), []).append((int(start), int(end)))
    for line in (labels_dir / "labels.jsonl").open(encoding="utf-8"):
        entry = json.loads(line)
        row = resolve_row(index, entry["recording_id"]) or rows.get(entry["recording_id"])
        if row is None:
            continue
        labels = np.load(labels_dir / entry["labels_path"])
        frame_ids = np.load(bundle / row["frame_ids_path"])
        boxes = (person_boxes_for(entry["recording_id"], person_boxes)
                 if person_boxes is not None else None)
        # A box array shorter than the frames it is indexed by would crop each
        # segment to some other moment of the video.  Refusing to crop is the
        # only safe answer; the count of refusals rides in the run's summary.
        if boxes is not None and len(frame_ids) and int(frame_ids.max()) >= len(boxes):
            boxes = None
        # One FK pass per recording, not per segment -- and only once some
        # segment of it is actually owned by this worker.
        joints: List[Optional[np.ndarray]] = [None]
        axis_angle: List[Optional[np.ndarray]] = [None]

        def joints_for() -> np.ndarray:
            if joints[0] is None:
                from tools.convert_motion_to_guofeats import motion_151_to_joints

                joints[0] = motion_151_to_joints(np.load(bundle / row["motion_path"]))
            return joints[0]

        def axis_angle_for() -> np.ndarray:
            # The released model takes rotations, and the bundle stores only the
            # packed 151-D, so this un-packs rather than re-reading the
            # converted directory -- which is a symlink into an evictable cache.
            if axis_angle[0] is None:
                from tools.posescript_capgen import motion_151_to_axis_angle

                axis_angle[0] = motion_151_to_axis_angle(
                    np.load(bundle / row["motion_path"]))
            return axis_angle[0]

        if spans is None:
            boundaries = segments_of(labels)
        else:
            # Clip to the label array rather than dropping the span, matching
            # recluster_atomics_ingroup.clip_span -- the segmentation's last
            # span for a recording can run a few frames past the motion array,
            # and the encoder that built this cache sliced with numpy, which
            # clips.  Dropping them here left 745 of aist_v1's M2-accepted
            # segments with no caption at all, so the w/ LLM re-clustering fell
            # back to the keyframe criterion on exactly those segments.  The
            # clipped span is a no-op for every span already inside the array,
            # so a caption file written before this change still resumes.
            boundaries = []
            for start, end in spans.get(entry["recording_id"], ()):
                end = min(int(end), len(labels))
                if end - int(start) < 1:
                    continue
                boundaries.append((int(start), end, int(labels[int(start)])))
        for start, end, label in boundaries:
            if label <= 0 or end - start < min_frames:
                continue
            key = (entry["recording_id"], start, end)
            mine = wanted is None or wanted(key)
            yield {
                "recording_id": entry["recording_id"],
                "start": start,
                "end": end,
                "prototype": int(label),
                "frame_ids": frame_ids,
                "split": row.get("split"),
                "boxes": boxes,
                "cue": (posescript_cue(
                            joints_for()[start:end], max_beats=max_beats,
                            describer=describer,
                            segment_axis_angle=(axis_angle_for()[start:end]
                                                if describer is not None else None))
                        if posescript and mine else ""),
            }


def video_path_for(recording_id: str, video_dir: pathlib.Path) -> Optional[pathlib.Path]:
    """The video a recording was captured in, for either corpus's id shape.

    Wild: ``tiktok:<upload>:<clip>`` -> ``<upload>__<clip>.mp4``.

    AIST++: ``aistpp/gBR_sBM_cAll_d04_mBR0_ch01`` -> the same name with a
    physical camera in place of ``cAll``.  The ``cAll`` token is AIST++'s
    annotation-space name -- the motion is reconstructed from all nine cameras
    and belongs to none of them -- so a file under that name never exists, and
    matching it literally is why this returned None for every AIST segment.
    The camera is resolved by glob rather than pinned to ``c01`` so that a
    corpus fetched from a different view still resolves, and the lowest name is
    taken when several are present, which keeps the choice deterministic.
    """
    parts = recording_id.split(":")
    if len(parts) >= 3:
        candidate = video_dir / "{}__{}.mp4".format(parts[-2], parts[-1])
        return candidate if candidate.exists() else None
    stem = recording_id.rsplit("/", 1)[-1]
    candidate = video_dir / "{}.mp4".format(stem)
    if candidate.exists():
        return candidate
    if "_cAll_" not in stem:
        return None
    found = sorted(video_dir.glob(stem.replace("_cAll_", "_c*_") + ".mp4"))
    return found[0] if found else None


def person_boxes_for(recording_id: str, root: pathlib.Path) -> Optional[np.ndarray]:
    """The tracked dancer's per-frame box, from whoever decided it.

    On the rebuilt corpus that is the ingest (``ingest_wild_uploads.py`` writes
    ``preprocess/bbx.pt`` per clip and GVHMR is seeded from it, so the two files
    agree by construction); on the older corpus it is GVHMR's own tracker.  The
    layout is the same either way, which is why one lookup serves both.

    On this corpus the dancer is a median 42% of frame height and under a third
    in 23% of clips, so a full frame spends most of its vision tokens on the
    studio.  Worse, wide shots contain a whole class: the caption then describes
    a crowd while the 3D motion it is attached to describes the one person
    GVHMR tracked, and those are different subjects.
    """
    parts = recording_id.split(":")
    if len(parts) < 3:
        return None
    path = root / "{}__{}".format(parts[-2], parts[-1]) / "preprocess" / "bbx.pt"
    if not path.is_file():
        return None
    import torch

    boxes = torch.load(path, map_location="cpu", weights_only=False)["bbx_xyxy"]
    return np.asarray(boxes, dtype=np.float64)


def segment_box(boxes: np.ndarray, positions: Sequence[int], *, margin: float,
                width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    """One crop for the whole segment: the union of its frames' boxes, expanded.

    Deliberately *not* a per-frame crop.  Re-centring the dancer every frame
    subtracts exactly the thing the caption is supposed to describe -- a step
    across the floor becomes a dancer standing still against a sliding
    background -- so the crop is fixed for the segment and travel stays visible
    inside it.
    """
    usable = [int(p) for p in positions if 0 <= int(p) < len(boxes)]
    if not usable:
        return None
    window = boxes[usable]
    x0, y0 = float(window[:, 0].min()), float(window[:, 1].min())
    x1, y1 = float(window[:, 2].max()), float(window[:, 3].max())
    if not (x1 > x0 and y1 > y0):
        return None
    pad_x, pad_y = (x1 - x0) * margin, (y1 - y0) * margin
    return (max(0, int(x0 - pad_x)), max(0, int(y0 - pad_y)),
            min(width, int(round(x1 + pad_x))), min(height, int(round(y1 + pad_y))))


def probe_frame_size(video: pathlib.Path) -> Tuple[int, int]:
    """Pixel size of a video, so a person box can be clamped to it."""
    import av

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        return int(stream.width), int(stream.height)


def load_frames(video: pathlib.Path, source_indices: Sequence[int], max_side: int,
                box: Optional[Tuple[int, int, int, int]] = None
                ) -> Optional[List["Image.Image"]]:  # noqa: F821
    import av
    from PIL import Image

    wanted = sorted(set(int(i) for i in source_indices))
    if not wanted:
        return None
    frames: Dict[int, "Image.Image"] = {}
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        target = set(wanted)
        for position, frame in enumerate(container.decode(stream)):
            if position in target:
                image = frame.to_image()
                if box is not None:
                    image = image.crop(box)
                scale = max_side / float(max(image.size))
                if scale < 1.0:
                    image = image.resize((max(1, int(image.width * scale)),
                                          max(1, int(image.height * scale))))
                frames[position] = image
                target.discard(position)
                if not target:
                    break
    # A short video can end before the last requested index; captioning what we
    # did get is better than dropping the segment, but fewer than two frames
    # cannot show a movement at all.
    got = [frames[i] for i in wanted if i in frames]
    return got if len(got) >= 2 else None


def check_video_frames_kept(inputs, frame_counts: Sequence[int]) -> None:
    """Refuse a video batch the processor quietly shrank.

    ``video_grid_thw`` rows are ``(t, h, w)`` and Qwen3-VL packs two frames per
    temporal position, so ``t`` must be ``ceil(frames / 2)``.  Anything smaller
    means the frames were resampled away, which is not visible in the reply and
    was not visible in any artifact this repo wrote: a caption produced from
    two frames looks exactly like a caption produced from thirty-two.
    """
    grid = inputs.get("video_grid_thw")
    if grid is None:
        raise CaptionError("processor returned no video_grid_thw; the video "
                           "path is not being used")
    rows = grid.tolist() if hasattr(grid, "tolist") else list(grid)
    if len(rows) != len(frame_counts):
        raise CaptionError("processor built {} video grid(s) for {} video(s)"
                           .format(len(rows), len(frame_counts)))
    for (temporal, _height, _width), count in zip(rows, frame_counts):
        expected = -(-count // 2)
        if int(temporal) != expected:
            raise CaptionError(
                "processor kept {} temporal position(s) for {} frame(s); "
                "expected {}.  The frames were resampled -- see "
                "Captioner.caption_video_batch".format(temporal, count, expected))


class Captioner:
    def __init__(self, model_dir: pathlib.Path, *, device: str, max_new_tokens: int,
                 prompt: Optional[str] = None):
        """``prompt`` overrides the closed-vocabulary caption instruction.

        The genre labeller asks the same model a different question over whole
        clips (``tools/label_clip_genre_vlm.py``), and everything else here --
        batched decode, the transformers dtype-keyword drift, left padding --
        is the same problem for both.  One argument is cheaper than a second
        copy of the loader that can drift from this one.
        """
        import torch
        from transformers import AutoConfig, AutoProcessor

        config = AutoConfig.from_pretrained(str(model_dir))
        architecture = (config.architectures or [""])[0]
        import transformers

        if not hasattr(transformers, architecture):
            raise CaptionError(
                "transformers {} has no {}; the installed version cannot load this "
                "model".format(transformers.__version__, architecture))
        loader = getattr(transformers, architecture)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(str(model_dir))
        # Batched generation requires left padding; see caption_batch.
        if getattr(self.processor, "tokenizer", None) is not None:
            self.processor.tokenizer.padding_side = "left"
        self.model = loader.from_pretrained(
            str(model_dir), device_map=device,
            **{dtype_kwarg(transformers.__version__): torch.bfloat16})
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt if prompt is not None else PROMPT.format(schema=schema_text())

    def caption_video_batch(self, batch: Sequence[Sequence["Image.Image"]],  # noqa: F821
                            cues: Optional[Sequence[str]] = None) -> List[str]:
        """Same prompt, but the frames go in as a *video* rather than N pictures.

        Qwen3-VL has a separate video path (``<|video_pad|>``, its own
        processor) that gives the frames temporal position encoding; as separate
        images they are an unordered set as far as the model's positions are
        concerned.  For a question whose answer lives in movement quality and
        timing -- dance genre -- that is not a cosmetic difference, and testing
        only the still-frame path would have condemned the model for a
        configuration rather than for the task.

        ``do_sample_frames=False`` is what makes that true, and until
        2026-08-22 it was missing.  ``Qwen3VLVideoProcessor`` defaults to
        ``fps=2`` and resamples whatever it is handed, so pre-sampled frames
        with no video metadata were silently reduced to two temporal positions
        no matter how many were passed.  Measured on this checkout:

            frames in   as shipped            with do_sample_frames=False
            6           grid t=2,   417 tok   grid t=3,    621 tok
            16          grid t=2,   417 tok   grid t=8,  1,641 tok
            32          grid t=2,   417 tok   grid t=16, 3,273 tok

        -- so the "video" arm was not merely weaker than the image arm, it was
        *smaller* than it (417 against 1,197 for six images), and 6 frames and
        32 frames produced byte-identical input sizes.  Every conclusion drawn
        from this path before that date was drawn from two frames.

        The packing is exactly ``t = ceil(frames / 2)``, so the assertion below
        can fail: it compares what the processor built against what was handed
        in, rather than trusting a keyword to have been honoured.
        """
        # The cue travels on this path too.  Without it the video arm differs
        # from the still arm in two ways at once, and a comparison between them
        # would not say which one moved the answer.
        cues = list(cues) if cues is not None else [""] * len(batch)
        texts = []
        videos = []
        for images, cue in zip(batch, cues):
            prompt = self.prompt + (CUE.format(cue=cue) if cue else "")
            messages = [{
                "role": "user",
                "content": [{"type": "video", "video": list(images)},
                            {"type": "text", "text": prompt}],
            }]
            texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
            videos.append(list(images))
        inputs = self.processor(text=texts, videos=videos, padding=True,
                                return_tensors="pt", do_sample_frames=False)
        check_video_frames_kept(inputs, [len(v) for v in videos])
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}
        with self.torch.no_grad():
            generated = self.model.generate(**inputs, do_sample=False,
                                            max_new_tokens=self.max_new_tokens)
        prompt_length = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(generated[:, prompt_length:],
                                           skip_special_tokens=True)

    def caption_batch(self, batch: Sequence[Sequence["Image.Image"]],  # noqa: F821
                      cues: Optional[Sequence[str]] = None) -> List[str]:
        """Caption several segments in one forward pass.

        Decoding is bound by streaming the weights, not by arithmetic, so a
        batch of eight costs barely more than a batch of one -- the difference
        between a two-day corpus run and a four-hour one.  Padding is on the
        left so that every sequence's generated tokens start at the same
        column and can be sliced uniformly; with right padding the slice would
        silently cut into pad tokens for the shorter prompts.
        """
        cues = list(cues) if cues is not None else [""] * len(batch)
        texts = []
        flat: List["Image.Image"] = []  # noqa: F821
        for images, cue in zip(batch, cues):
            prompt = self.prompt + (CUE.format(cue=cue) if cue else "")
            messages = [{
                "role": "user",
                "content": [{"type": "image", "image": image} for image in images]
                           + [{"type": "text", "text": prompt}],
            }]
            texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
            flat.extend(images)
        inputs = self.processor(text=texts, images=flat, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None)
        width = inputs["input_ids"].shape[1]
        return [self.processor.decode(row[width:], skip_special_tokens=True)
                for row in generated]

    def caption(self, images: Sequence["Image.Image"], cue: str = "") -> str:  # noqa: F821
        return self.caption_batch([images], [cue])[0]


def run(*, labels_dir: pathlib.Path, bundle: pathlib.Path, video_dir: pathlib.Path,
        model_dir: pathlib.Path, output: pathlib.Path, shard: int, num_shards: int,
        frames_per_segment: int, max_side: int, min_frames: int, limit: Optional[int],
        device: str, max_new_tokens: int, dry_run: bool,
        batch_size: int = 1, posescript: bool = False, max_beats: int = 4,
        person_boxes: Optional[pathlib.Path] = None,
        crop_margin: float = 0.25,
        posescript_model: Optional[pathlib.Path] = None,
        embedding_cache: Optional[pathlib.Path] = None,
        video_per_motion: int = 1,
        as_video: bool = False) -> Dict[str, object]:
    if person_boxes is not None and video_per_motion != 1:
        # The boxes are indexed by source frame and the guard in iter_segments
        # compares them against unscaled frame ids, so the pair would crop each
        # segment to some other moment of the video.  Nothing downstream could
        # tell: the caption would describe a real dancer doing something else.
        raise CaptionError("--person-boxes and --video-per-motion {} cannot be "
                           "combined; the boxes are not on that frame grid"
                           .format(video_per_motion))
    output.parent.mkdir(parents=True, exist_ok=True)

    # Resume: a captioning run over ~20k segments will be interrupted, and
    # re-captioning what is already done wastes the GPU and, worse, can produce
    # a second differing caption for the same segment.
    done = set()
    if output.exists():
        for line in output.open(encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((row.get("recording_id"), row.get("start"), row.get("end")))

    captioner = None if dry_run else Captioner(
        model_dir, device=device, max_new_tokens=max_new_tokens)

    # The cue's author is provenance, not a detail: "PoseScript said so" and
    # "our posecode rules said so" are different claims and land in every row.
    describer = None
    cue_source = "none"
    if posescript:
        cue_source = "motion_beats.describe_pose(rule-based)"
        if posescript_model is not None:
            from tools.posescript_capgen import PoseScriptCaptioner

            describer = PoseScriptCaptioner(posescript_model, device=device)
            cue_source = "posescript:{}".format(pathlib.Path(describer.checkpoint).parts[-3])

    counts = {"seen": 0, "mine": 0, "skipped_done": 0, "no_video": 0,
              "no_frames": 0, "unparsed": 0, "written": 0,
              "cropped": 0, "uncropped": 0}
    frame_sizes: Dict[pathlib.Path, Tuple[int, int]] = {}
    started = time.time()
    pending: List[Tuple[Dict[str, object], List[object]]] = []

    with output.open("a", encoding="utf-8") as handle:

        def flush_batch() -> None:
            """Caption everything queued and write one row per segment.

            Rows are written only after generation succeeds for the batch, so
            an interrupted run leaves no half-written row for the resume scan
            to mistake for finished work.
            """
            if not pending:
                return
            send = (captioner.caption_video_batch if as_video
                    else captioner.caption_batch)
            replies = send(
                [images for _, images in pending],
                [str(segment.get("cue", "")) for segment, _ in pending])
            for (segment, images), reply in zip(pending, replies):
                parsed = parse_caption(reply)
                if parsed is None:
                    counts["unparsed"] += 1
                    continue
                handle.write(json.dumps({
                    "schema_version": _schema["version"],
                    "input_mode": "video" if as_video else "images",
                    "producer_version": PRODUCER_VERSION,
                    "model": model_name(model_dir),
                    "substitutes_for": "gemini-2.5-pro (paper M3); recorded substitution",
                    "recording_id": segment["recording_id"],
                    "start": int(segment["start"]),
                    "end": int(segment["end"]),
                    "prototype": int(segment["prototype"]),
                    "split": segment.get("split"),
                    "frames_used": len(images),
                    "posescript_cue": segment.get("cue") or None,
                    "posescript_source": cue_source,
                    "fields": {field: parsed[field]
                               for field in _schema["fields"]},
                    "summary": parsed["summary"],
                    "caption": caption_sentence(parsed),
                }, sort_keys=True) + "\n")
                counts["written"] += 1
            handle.flush()
            pending.clear()
            rate = counts["written"] / max(1e-6, time.time() - started)
            print("  shard {}: {} captions, {:.2f}/s".format(
                shard, counts["written"], rate), flush=True)

        for segment in iter_segments(
                labels_dir, bundle, min_frames=min_frames, posescript=posescript,
                max_beats=max_beats, person_boxes=person_boxes,
                describer=describer, embedding_cache=embedding_cache,
                wanted=lambda key: shard_of(key, num_shards) == shard):
            counts["seen"] += 1
            key = (segment["recording_id"], segment["start"], segment["end"])
            if shard_of(key, num_shards) != shard:
                continue
            counts["mine"] += 1
            if key in done:
                counts["skipped_done"] += 1
                continue
            if limit is not None and counts["written"] + len(pending) >= limit:
                break
            video = video_path_for(str(segment["recording_id"]), video_dir)
            if video is None:
                counts["no_video"] += 1
                continue
            positions = sample_frame_positions(
                int(segment["start"]), int(segment["end"]), frames_per_segment)
            frame_ids = segment["frame_ids"]
            source = [int(frame_ids[p]) * video_per_motion
                      for p in positions if p < len(frame_ids)]
            box = None
            if segment.get("boxes") is not None:
                # Sizes are per video and the boxes are in its pixel space, so
                # they are cached per path rather than once for the run.
                if video not in frame_sizes:
                    frame_sizes[video] = probe_frame_size(video)
                width, height = frame_sizes[video]
                box = segment_box(segment["boxes"], source, margin=crop_margin,
                                  width=width, height=height)
            counts["cropped" if box is not None else "uncropped"] += 1
            if dry_run:
                counts["written"] += 1
                continue
            images = load_frames(video, source, max_side, box)
            if images is None:
                counts["no_frames"] += 1
                continue
            pending.append((segment, images))
            if len(pending) >= batch_size:
                flush_batch()

        flush_batch()

    counts["elapsed_s"] = round(time.time() - started, 1)
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=pathlib.Path, required=True)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--video-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--frames-per-segment", type=int, default=6)
    parser.add_argument("--max-side", type=int, default=448)
    parser.add_argument("--min-frames", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="segments per forward pass; decode is weight-bound, "
                             "so this is close to a linear speed-up")
    parser.add_argument("--schema", choices=sorted(SCHEMAS), default="v1",
                        help="closed vocabulary to fill.  v1 is the six axes "
                             "the published corpus was captioned with; v2 adds "
                             "rhythm and splits v1's single dynamics axis into "
                             "intensity and fluidity, which is what the paper "
                             "names as the components of movement dynamics")
    parser.add_argument("--as-video", action="store_true",
                        help="feed the frames through Qwen3-VL's video path, "
                             "which gives them temporal positions; the still "
                             "path presents them as an unordered set")
    parser.add_argument("--posescript", action="store_true",
                        help="attach PoseScript-style motion-beat pose sentences to the "
                             "prompt, the paper's auxiliary cue to the VLM descriptor")
    parser.add_argument("--posescript-model", type=pathlib.Path, default=None,
                        help="PoseScript's released capgen checkpoint; without it the "
                             "cue comes from the rule-based fallback, and either way "
                             "the choice is recorded in every caption row")
    parser.add_argument("--max-beats", type=int, default=4,
                        help="motion beats described in the cue")
    parser.add_argument("--person-boxes", type=pathlib.Path, default=None,
                        help="GVHMR raw output root; crops each segment to the tracked "
                             "dancer instead of captioning the whole frame")
    parser.add_argument("--crop-margin", type=float, default=0.25,
                        help="fraction of the box added on each side")
    parser.add_argument("--embedding-cache", type=pathlib.Path, default=None,
                        help="the segment-level cache M2 clustered. Supplies the "
                             "segment boundaries, so captions are keyed to the same "
                             "segments tools/recluster_atomics_ingroup.py re-clusters "
                             "with the same flag. Without it both stages fall back to "
                             "runs of equal frame label, and mixing the two produces "
                             "captions that cannot be matched at all.")
    parser.add_argument("--video-per-motion", type=int, default=1,
                        help="source video frames per motion frame: 1 for the wild "
                             "corpus, whose frame_ids.npy already indexes the source, "
                             "and 2 for AIST++, whose 30 fps motion frame t comes from "
                             "video frame 2t (the same 2:1 tools/extract_visual_"
                             "features.py uses). Getting it wrong reads the right "
                             "video at the wrong moment, which nothing downstream "
                             "can detect.")
    parser.add_argument("--dry-run", action="store_true",
                        help="walk segments and frame selection without loading the model")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.shard < 0 or args.shard >= args.num_shards:
        raise SystemExit("error: --shard must be in [0, --num-shards)")
    # Set before the prompt is built: the prompt is the schema.
    use_schema(args.schema)
    try:
        counts = run(labels_dir=args.labels, bundle=args.bundle, video_dir=args.video_dir,
                     model_dir=args.model, output=args.output, shard=args.shard,
                     num_shards=args.num_shards, frames_per_segment=args.frames_per_segment,
                     max_side=args.max_side, min_frames=args.min_frames, limit=args.limit,
                     device=args.device, max_new_tokens=args.max_new_tokens,
                     dry_run=args.dry_run, batch_size=args.batch_size,
                     posescript=args.posescript, max_beats=args.max_beats,
                     person_boxes=args.person_boxes, crop_margin=args.crop_margin,
                     posescript_model=args.posescript_model,
                     embedding_cache=args.embedding_cache,
                     video_per_motion=args.video_per_motion,
                     as_video=args.as_video)
    except (CaptionError, FileNotFoundError) as error:
        raise SystemExit("error: {}".format(error))
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
