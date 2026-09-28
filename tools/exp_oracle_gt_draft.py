#!/usr/bin/env python3
"""COMPLETION ORACLE: hand the completion the clip's OWN ground truth as the draft.

WHY THIS EXISTS.  docs/DANCE_QUALITY_DEFECTS.md section 21 ran the PLAN oracle
(ground-truth atomic labels into the shipping pipeline) and found that a perfect
plan does not fix the completion.  The remaining oracle is a perfect DRAFT.  The
draft-only measurement (runs/opt_draftonly) shows the retrieval draft already
carries stillness -- sustained-hold share 0.0411 against a ground truth of
0.0176 -- and full travel (root_vs_gt 1.0718), while the completion's OUTPUT
reads sustained-hold share **exactly 0.0000** and root_vs_gt 0.6193.  So the two
readings this experiment separates are:

  * the completion, handed the clip's own ground truth as its draft, still
    outputs a body with no sustained stillness and ~60% of the travel
    -> the defect is INSIDE the completion, regardless of input, and no
    retrieval-side or planner-side change can reach it;
  * it reproduces the ground truth closely
    -> the defect is the distribution shift between TRAINING drafts (another
    performance of the same classes, whose timing is uncorrelated with the
    target by construction -- see dataset/atomic.py's ``warp_to_anchors``
    docstring, which measured output-vs-own-draft speed correlation -0.110) and
    INFERENCE drafts, and the fix belongs on the draft side.

ADD-ONLY.  Nothing in infer_atomic.py is modified.  Everything that decides what
the output is -- ``infer_completion`` (windowing AND the overlap-add stitch),
``_fill_draft_gaps``, ``_blend_draft_seams``, ``_write_generated_result``,
``sample_seed``, ``_load_music``, ``_load_checkpoint`` -- is imported from it and
called, so a divergence from the shipping arm cannot hide in a second code path.
In particular the crossfade is NOT reimplemented here: the built-in
``--ground-truth-labels`` path refuses any clip longer than one 150-frame slice
("ORACLE ground-truth-plan inference currently supports at most one 150-frame
slice"), so this file does the *driving* of the whole-clip loop itself and hands
every window to ``infer_completion``, which is the shipping stitch.

THE THREE ARMS, and what each one alone can say.

``gt``        draft = the clip's own ground-truth 151-D motion, mask = 1 on
              every frame.  The purest oracle: the model is told "this draft is
              retrieved everywhere and it is trusted".
``gt_nofill`` draft = ground truth everywhere, mask = 0 on the frames the
              ground-truth plan calls label 0.  Same VALUES as ``gt``; only the
              mask (which is the model's noise scale AND its declaration of what
              was retrieved) takes the shipping pattern.  Isolates the mask.
``gt_ops``    the shipping draft OPERATIONS applied to a ground-truth draft:
              label-0 frames zeroed and mask 0 there, then
              ``_fill_draft_gaps(..., "interpolate")``, then
              ``_blend_draft_seams(..., half_width=4)`` -- the same two calls in
              the same order ``IndexedAtomicMotionLibrary.build_draft`` makes
              them, with the shipping arm's ``--draft-gap-fill interpolate
              --draft-seam-blend 4``.  ``gt_nofill`` vs ``gt_ops`` is therefore
              exactly "do the gap-fill ramps destroy the stillness", with the
              mask held fixed between them.

WHERE THE GROUND-TRUTH DRAFT COMES FROM, and why it is not the eval pkls.  The
scoring pkls under ``runs/txy_t_gt_eval/motion`` carry ``full_pose`` only --
joint POSITIONS after forward kinematics -- and the completion consumes
normalized 151-D [4 contacts, 3 root, 144 rot6d].  Inverting FK is not exact, so
the draft is read from the release instead, by the same reconstruction
``GroundTruthPlanStore`` uses for labels: ``windows.jsonl``'s own
``start_frame``/``end_frame_exclusive`` per window, overlapping windows written
over each other, and the redundancy kept as the check rather than discarded.
Measured on the 18 test clips the release covers: **0.0 conflict** between
overlapping windows and 0 uncovered frames.

TWO CLIPS ARE NOT IN THE RELEASE AT ALL.
``wild_v5:7188505181892381984:clip000`` and ``wild_v5:7610414564962183545:clip000``
appear in no split of ``windows.jsonl`` (and ``quarantine.jsonl`` is empty), so
they have no release-space ground-truth motion and no ground-truth labels.  For
those two the draft is rebuilt from the per-sequence RAW 151-D array
(``data/wild3d/txy_t_normalized/sequences_normalized.jsonl`` ->
``assets.motion_151_raw``) put through the release's own ``normalizer.pt`` with
the release's documented formula.  That path is not assumed, it is CALIBRATED:
on the 18 clips where both sources exist it reproduces the release windows on
**9649 of 9885 frames to float32 rounding (median |delta| 1.9e-7)**, with 236
frames (2.4%) differing and a worst-case 0.279 in normalized units.  The same
frames-beyond-release-coverage rule fills each clip's tail, because the release
truncates a clip to a whole number of window strides (e.g. 615 covered frames
against 619 music frames) while the generation length is the MUSIC length.

Consequence for the arms: ``gt`` runs on all 20 test clips (it needs no labels);
``gt_nofill`` and ``gt_ops`` need the ground-truth label track and therefore run
on the 18 the release covers.  Stated here rather than left to be inferred from
a clip count.

SEEDING.  ``seed_everything(sample_seed(seed, name))`` is called immediately
before each clip's ``infer_completion``, so the three arms here are noise-PAIRED
per clip with each other.  They are NOT noise-paired with the shipping arm: in
``infer_directory`` the planner sample and the retrieval draft consume draws
between the seeding and the completion, and this file makes neither of those
calls.  Comparisons against runs/txy_t_m6_ep12 are therefore aggregate over the
clip set, at the same base seed and the same sampler settings, not per-clip
paired -- the same distinction the shipping manifest's ``seed_pairing`` key
exists to record.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import infer_atomic as IA  # noqa: E402

RELEASE = "/cache/atomicdance-assets/scratch/txy_t/release_v3"
SHIPPING_COMPLETION = "runs/completion_txy_t_long_s20260902/completion_step50400.pt"
SHIPPING_PLANNER = "runs/planner_txy_t_short_s20260902/planner_epoch12_step1008.pt"
SEQUENCE_MANIFEST = "data/wild3d/txy_t_normalized/sequences_normalized.jsonl"
ARMS = ("gt", "gt_nofill", "gt_ops")


def release_windows(data_root, split=None):
    """sequence_id -> sorted [(start, end_exclusive, array_index, split)].

    The same source ``GroundTruthPlanStore`` reads; kept separate because that
    class indexes LABELS and this needs the motion array's window spans.
    """
    spans = {}
    manifest = pathlib.Path(data_root) / "windows.jsonl"
    with open(str(manifest)) as handle:
        for line in handle:
            row = json.loads(line)
            if split is not None and row.get("split") != split:
                continue
            spans.setdefault(row["sequence_id"], []).append(
                (int(row["start_frame"]), int(row["end_frame_exclusive"]),
                 int(row["array_index"]), row.get("split")))
    for value in spans.values():
        value.sort()
    return spans


def sequence_records(manifest_path=SEQUENCE_MANIFEST):
    records = {}
    with open(str(manifest_path)) as handle:
        for line in handle:
            row = json.loads(line)
            records[row["sequence_id"]] = row
    return records


def normalizer_affine(data_root):
    """(data_min, safe_range) for the release's own min-max, as float64.

    Deliberately the same ``safe_range`` rule ``infer_atomic.unnormalize_motion``
    inverts: an exactly constant dimension gets a unit range, a tiny but
    non-zero one stays a range.
    """
    state = torch.load(str(pathlib.Path(data_root) / "normalizer.pt"),
                       map_location="cpu", weights_only=False)
    minimum = state["data_min"].float().numpy().astype(np.float64)
    maximum = state["data_max"].float().numpy().astype(np.float64)
    return minimum, np.where(maximum == minimum, 1.0, maximum - minimum)


class GroundTruthMotionStore:
    """The clip's own normalized 151-D motion, release-first, raw-renormalized second.

    ``get(name, frames)`` returns ``(motion[frames, 151], provenance)`` where
    provenance counts how many frames came from each of the two sources, so an
    artifact can never be read without knowing which one produced it.
    """

    def __init__(self, data_root=RELEASE, sequence_manifest=SEQUENCE_MANIFEST):
        self.data_root = pathlib.Path(data_root)
        self.spans = release_windows(self.data_root)
        self.records = sequence_records(sequence_manifest)
        self.minimum, self.safe_range = normalizer_affine(self.data_root)
        self._arrays = {}
        self.conflicts = 0.0

    def _split_array(self, split):
        if split not in self._arrays:
            self._arrays[split] = np.load(
                str(self.data_root / split / "motion.npy"), mmap_mode="r")
        return self._arrays[split]

    def _raw_renormalized(self, name):
        record = self.records[name]
        raw = np.load(record["assets"]["motion_151_raw"]).astype(np.float64)
        return (2.0 * (raw - self.minimum) / self.safe_range - 1.0).astype(np.float32)

    def get(self, name, frames):
        fallback = self._raw_renormalized(name)
        if frames > len(fallback):
            raise ValueError(
                "ground-truth motion for {} covers {} frames, {} requested".format(
                    name, len(fallback), frames))
        track = np.array(fallback[:frames], dtype=np.float32, copy=True)
        from_release = np.zeros(frames, bool)
        if name in self.spans:
            for start, end, index, split in self.spans[name]:
                if start >= frames:
                    continue
                stop = min(end, frames)
                values = np.asarray(
                    self._split_array(split)[index][: end - start][: stop - start],
                    np.float32)
                written = from_release[start:stop]
                if written.any():
                    self.conflicts = max(self.conflicts, float(
                        np.abs(track[start:stop][written] - values[written]).max()))
                track[start:stop] = values
                from_release[start:stop] = True
        return torch.from_numpy(track), {
            "frames": int(frames),
            "frames_from_release_windows": int(from_release.sum()),
            "frames_from_raw_renormalized": int(frames - from_release.sum()),
            "in_release": bool(name in self.spans),
        }


def build_conditions(motion, labels, arm, *, seam_blend=4, gap_fill="interpolate"):
    """(draft, mask) for one arm.  ``labels`` may be None only for arm ``gt``.

    The two draft operations are ``infer_atomic``'s own functions, called in the
    order ``build_draft`` calls them (gaps first, seams second), so ``gt_ops``
    is the shipping treatment and not an imitation of it.
    """
    frames = motion.shape[0]
    draft = motion.clone()
    if arm == "gt":
        return draft, torch.ones(frames, 1, dtype=torch.float32)
    if labels is None:
        raise ValueError("arm {!r} needs the ground-truth label track".format(arm))
    if len(labels) != frames:
        raise ValueError("labels cover {} frames, motion {}".format(len(labels), frames))
    mask = (labels != 0).to(torch.float32).unsqueeze(1)
    if arm == "gt_nofill":
        return draft, mask
    if arm != "gt_ops":
        raise ValueError("unknown arm {!r}".format(arm))
    # The shipping draft never contains a value for a label-0 frame: the
    # retrieval loop skips those segments and leaves the zeros build_draft
    # allocated.  Reproduce that BEFORE the two operations, or the gap fill has
    # nothing to fill and the arm is silently gt_nofill.
    draft[mask[:, 0] == 0] = 0.0
    IA._fill_draft_gaps(draft, mask, gap_fill)
    if seam_blend:
        IA._blend_draft_seams(draft, mask, labels, seam_blend)
    return draft, mask


def label_track(plans, name, frames):
    """The ground-truth label track padded to ``frames`` with 0 (transition).

    The release truncates a clip to a whole number of window strides while the
    generation length is the music length, so the last few frames of most clips
    have no ground-truth label.  Padding with 0 is the same convention
    ``infer_completion`` uses for a short tail window ("nothing planned here"),
    and the count of padded frames rides into the manifest rather than being
    left to be inferred.
    """
    if plans is None or not plans.has_sequence(name):
        return None, 0
    track = plans._full_track(name)
    covered = int(min(len(track), frames))
    if (track[:covered] < 0).any():
        raise ValueError("ground-truth labels for {} have holes".format(name))
    padded = np.zeros(frames, np.int64)
    padded[:covered] = track[:covered]
    return torch.from_numpy(padded), int(frames - covered)


def run_arm(arm, clips, *, audio_dir, output_dir, data_root, completion_checkpoint,
            seed, guidance_weight, stride, blend_width, batch_size, device,
            motions, plans, overwrite=False, drafts_only=False):
    """``drafts_only`` writes what the model was HANDED, not what it produced.

    Written through ``_write_generated_result`` -- the same writer, from the same
    tensor object the sampler would receive -- so the draft reaches every scorer
    in the same space as a real arm.  Without it, "the gap-fill ramps destroyed
    the stillness" and "the model destroyed it" are indistinguishable from the
    output alone; this is ``infer_directory``'s ``--draft-dump-dir`` argument
    made available for a draft the shipping path cannot build.
    """
    device = IA.resolve_device("cpu" if drafts_only else device)
    provenance = IA.validate_training_data_root(data_root)
    completion, completion_args = IA._load_checkpoint(
        completion_checkpoint, "completion", device)
    if getattr(getattr(completion, "model", None), "label_embedding", None) is not None:
        raise ValueError(
            "this completion has a label channel; the shipping arm's does not, and "
            "conditioning on labels here would change more than the draft")
    ratio = completion_args.draft_noise_ratio
    normalizer_path = pathlib.Path(data_root) / "normalizer.pt"
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    skipped = []
    for name in clips:
        output_path = output_dir / (name + ".pkl")
        audio_path = pathlib.Path(audio_dir) / (name + ".npy")
        music = IA._load_music(audio_path, None, completion_args.music_dim)
        frames = len(music)
        labels, label_pad = label_track(plans, name, frames)
        if arm != "gt" and labels is None:
            skipped.append(name)
            continue
        motion, motion_provenance = motions.get(name, frames)
        draft, mask = build_conditions(motion, labels, arm)
        clip_seed = IA.sample_seed(seed, name)
        if output_path.is_file() and not overwrite:
            rows[name] = {"reused": True}
            continue
        IA.seed_everything(clip_seed)
        if drafts_only:
            normalized = draft
        else:
            normalized = IA.infer_completion(
                completion,
                music,
                draft,
                mask * ratio,
                completion_args.seq_len,
                stride,
                device,
                guidance_weight,
                batch_size,
                labels=None,
                blend_width=blend_width,
            )
        written_labels = (labels if labels is not None
                          else torch.zeros(frames, dtype=torch.int64))
        IA._write_generated_result(
            output_path,
            normalized,
            written_labels,
            audio_path,
            normalizer_path,
            SHIPPING_PLANNER,
            completion_checkpoint,
            generation_protocol="ORACLE_GROUND_TRUTH_{}_{}".format(
                "DRAFT_INPUT" if drafts_only else "DRAFT", arm.upper()),
            headline_eligible=False,
            query_retrieval_group_id=None,
            safe_draft_condition_fraction=float(mask.mean()),
            plan_atomic_segments=(IA._plan_atomic_segments(written_labels)
                                  if labels is not None else None),
            target_motion_selection="ORACLE_GROUND_TRUTH_MOTION_AS_DRAFT",
            dataset_provenance=provenance,
        )
        rows[name] = {
            "frames": frames,
            "seed": int(clip_seed),
            "draft_conditioned_fraction": float(mask.mean()),
            "label_frames_padded_with_transition": label_pad,
            "motion_provenance": motion_provenance,
            "draft_vs_gt_max_abs": float((draft - motion).abs().max()),
        }
    manifest = {
        "arm": arm,
        "experiment": ("COMPLETION_ORACLE_GROUND_TRUTH_DRAFT_INPUT" if drafts_only
                       else "COMPLETION_ORACLE_GROUND_TRUTH_DRAFT"),
        "is_the_model_input_not_its_output": bool(drafts_only),
        "headline_eligible": False,
        "clips": len(rows),
        "skipped_no_ground_truth_labels": skipped,
        "output_dir": str(output_dir),
        "completion_checkpoint": str(completion_checkpoint),
        "data_root": str(data_root),
        "audio_dir": str(audio_dir),
        "sampling": {
            "seed": seed,
            "guidance_weight": guidance_weight,
            "completion_stride": stride,
            "completion_blend_width": blend_width,
            "inference_batch_size": batch_size,
            "draft_noise_ratio": ratio,
            "seq_len": completion_args.seq_len,
            "seed_pairing": "per-clip-across-oracle-arms-only",
            "draft_gap_fill": "interpolate" if arm == "gt_ops" else "none",
            "draft_seam_blend": 4 if arm == "gt_ops" else 0,
            "mask_policy": {"gt": "ones", "gt_nofill": "labels!=0",
                            "gt_ops": "labels!=0"}[arm],
        },
        "ground_truth_motion_window_conflict_max": motions.conflicts,
        "per_clip": rows,
        "code_revision": IA._code_revision(),
    }
    with open(str(output_dir / "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", choices=ARMS, required=True)
    parser.add_argument("--clips", required=True, type=pathlib.Path)
    parser.add_argument("--audio-dir", default="runs/txy_t_gt_eval/audio")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--data-root", default=RELEASE)
    parser.add_argument("--completion-checkpoint", default=SHIPPING_COMPLETION)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--guidance-weight", type=float, default=2.0)
    parser.add_argument("--completion-stride", type=int, default=75)
    parser.add_argument("--completion-blend-width", type=int, default=10)
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--drafts-only", action="store_true",
                        help="write what the model would be HANDED, not what it "
                             "produces; no GPU, output under runs/opt_oracle_draft_<arm>")
    args = parser.parse_args()

    clips = [line.strip() for line in args.clips.read_text().splitlines() if line.strip()]
    motions = GroundTruthMotionStore(args.data_root)
    plans = IA.GroundTruthPlanStore(args.data_root)
    for arm in args.arm:
        prefix = "opt_oracle_draft_" if args.drafts_only else "opt_oracle_"
        output_dir = pathlib.Path(args.output_root) / (prefix + arm)
        manifest = run_arm(
            arm, clips, audio_dir=args.audio_dir, output_dir=output_dir,
            data_root=args.data_root, completion_checkpoint=args.completion_checkpoint,
            seed=args.seed, guidance_weight=args.guidance_weight,
            stride=args.completion_stride, blend_width=args.completion_blend_width,
            batch_size=args.inference_batch_size, device=args.device,
            motions=motions, plans=plans, overwrite=args.overwrite,
            drafts_only=args.drafts_only)
        print("{}: {} clips -> {} (skipped {})".format(
            arm, manifest["clips"], output_dir,
            len(manifest["skipped_no_ground_truth_labels"])))


if __name__ == "__main__":
    main()
