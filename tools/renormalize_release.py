#!/usr/bin/env python3
"""Republish a materialized release under a robust (percentile) normalizer.

WHAT PROBLEM.  ``tools/fit_motion_normalizer.py`` fits a global per-dimension
min-max over the train split, with no clipping, on purpose -- "do not add an
epsilon or otherwise 'fix' constant dimensions".  On the wild corpus that makes
the two root translation channels hostage to a handful of clips whose monocular
odometry wandered.  Read straight out of the shipped ``normalizer.pt``:

    dim 4 (root x)   -6.9723 .. 11.6300     range 18.6023 m
    dim 5 (root y)  -15.3334 .. 18.4958     range 33.8292 m
    dim 6 (root z)    0.3885 ..  3.4674     range  3.0789 m

while the corpus's own p1-p99 spans are 2.92 and 3.61 m.  After normalization a
150-frame window's horizontal root moves 0.048 / 0.028 normalized units against
a per-dimension rot6d standard deviation of 0.191, so the root occupies roughly
4% of the range it is scored against -- **0.117% of the denoising MSE for 3 of
151 dimensions** -- and a model that simply predicts the mean is near-optimal
there.  The measured consequence is D3: max horizontal root displacement 0.256 m
against the ground truth's 1.206 m, with path length 1.29-1.35x, i.e. jittering
in place rather than travelling.

WHY THIS TOOL AND NOT A REBUILD FROM SOURCE.  Normalization is a per-dimension
affine map, so re-normalizing an already-normalized array is the composition of
two affines and is exact -- no decode, no re-slicing, no re-derivation of
windows, and none of the 25 GB source pipeline.  ``old -> raw -> new`` is

    raw = (old + 1) / 2 * old_range + old_min
    new = 2 * (raw - new_min) / new_range - 1

which collapses to ``new = scale * old + shift`` with

    scale = old_range / new_range
    shift = (2 * (old_min - new_min) + old_range) / new_range - 1

The composition is verified, not asserted: a random sample of frames is decoded
through both normalizers and the raw values must agree to 1e-4 m before
anything is published.  That check is the point of the tool -- an affine that is
off by a factor reads as a perfectly plausible corpus.

WHAT IT DOES NOT CHANGE.  Only the normalizer and ``motion.npy``.  Labels,
music, masks, names and retrieval groups are hardlinked, so the release is the
same windows over the same frames; the contract in ``build.json`` is rewritten
with the new hashes and an added ``derived_from`` block naming the source
release and the percentiles, so nothing has to be taken on trust.

CLIPPING.  Frames outside the new range clip to +-1 on those dimensions.  At
p0.1/p99.9 that is 0.200% of training targets, and they are the odometry
excursions the fit is trying not to be dominated by; the reverse-diffusion
iterate is already clamped to [-1, 1] (``model/atomic_completion.py``), so the
model could never have represented them anyway.  The clipped fraction is
measured and written into the report rather than assumed.
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil

import numpy as np
import torch

MOTION_DIM = 151
ROOT_DIMS = (4, 5, 6)
COPY_FILES = ("labels.npy", "music.npy", "label_valid_mask.npy",
              "names.json", "retrieval_groups.json")


def sha256_file(path, block=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_normalizer(path):
    payload = torch.load(str(path), map_location="cpu")
    return (np.asarray(payload["data_min"], np.float64),
            np.asarray(payload["data_max"], np.float64))


def raw_percentiles(motion_path, data_min, data_max, dims, low, high, block=2048):
    """Percentiles of the RAW values, streamed, on the dimensions being refit."""
    array = np.load(motion_path, mmap_mode="r")
    span = np.maximum(data_max - data_min, 1e-12)
    samples = []
    for start in range(0, len(array), block):
        chunk = np.asarray(array[start:start + block, :, dims], np.float64)
        raw = (chunk + 1.0) / 2.0 * span[list(dims)] + data_min[list(dims)]
        samples.append(raw.reshape(-1, len(dims)))
    stacked = np.concatenate(samples, 0)
    return (np.percentile(stacked, low, axis=0), np.percentile(stacked, high, axis=0),
            len(stacked))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--low", type=float, default=0.1)
    parser.add_argument("--high", type=float, default=99.9)
    parser.add_argument("--dims", type=int, nargs="+", default=list(ROOT_DIMS))
    parser.add_argument("--check-frames", type=int, default=4096)
    parser.add_argument("--target-normalizer", default=None,
                        help="re-express the release in THIS normalizer's units (all dims) "
                             "instead of refitting percentiles.  Why: a corpus that grows by "
                             "a few clips would otherwise get new units, and every checkpoint "
                             "and selector trained in the old ones -- nothing at inference "
                             "compares the two -- would read the new library wrongly.  The "
                             "published normalizer.pt is a byte copy of the target.")
    args = parser.parse_args()

    source = pathlib.Path(args.source)
    output = pathlib.Path(args.output)
    if output.exists():
        raise SystemExit("{} exists; releases are immutable".format(output))
    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)

    data_min, data_max = load_normalizer(source / "normalizer.pt")
    if args.target_normalizer:
        # Every dimension changes units here, so every dimension is measured.
        dims = list(range(MOTION_DIM))
        new_min, new_max = load_normalizer(args.target_normalizer)
        frames = 0
    else:
        dims = list(args.dims)
        low, high, frames = raw_percentiles(source / "train" / "motion.npy",
                                            data_min, data_max, dims, args.low, args.high)
        new_min, new_max = data_min.copy(), data_max.copy()
        new_min[dims], new_max[dims] = low, high

    old_range = np.maximum(data_max - data_min, 1e-12)
    new_range = np.maximum(new_max - new_min, 1e-12)
    scale = old_range / new_range
    shift = (2.0 * (data_min - new_min) + old_range) / new_range - 1.0

    print("target normalizer {}".format(args.target_normalizer) if args.target_normalizer
          else "refit on {} frames".format(frames))
    for dim in (ROOT_DIMS if args.target_normalizer else dims):
        print("  dim {:>3}: {:.4f}..{:.4f} ({:.4f} m) -> {:.4f}..{:.4f} ({:.4f} m), {:.2f}x narrower"
              .format(dim, data_min[dim], data_max[dim], old_range[dim],
                      new_min[dim], new_max[dim], new_range[dim],
                      old_range[dim] / new_range[dim]))

    staging.mkdir(parents=True)
    clipped_total = clipped_count = 0
    for split in ("train", "val", "test"):
        source_split = source / split
        if not source_split.is_dir():
            continue
        target_split = staging / split
        target_split.mkdir()
        for name in COPY_FILES:
            if (source_split / name).is_file():
                os.link(source_split / name, target_split / name)
        array = np.load(source_split / "motion.npy", mmap_mode="r")
        out = np.lib.format.open_memmap(target_split / "motion.npy", mode="w+",
                                        dtype=array.dtype, shape=array.shape)
        for start in range(0, len(array), 2048):
            chunk = np.asarray(array[start:start + 2048], np.float64)
            mapped = chunk * scale + shift
            clipped_total += mapped[..., dims].size
            clipped_count += int(np.count_nonzero(np.abs(mapped[..., dims]) > 1.0))
            out[start:start + 2048] = np.clip(mapped, -1.0, 1.0).astype(array.dtype)
        out.flush()
        del out

        # Verify the composition on real rows rather than trusting the algebra.
        rows = np.random.default_rng(0).choice(len(array),
                                               size=min(args.check_frames, len(array)),
                                               replace=False)
        before = np.asarray(array[np.sort(rows)], np.float64)
        after = np.asarray(np.load(target_split / "motion.npy", mmap_mode="r")[np.sort(rows)],
                           np.float64)
        raw_before = (before + 1) / 2 * old_range + data_min
        raw_after = (after + 1) / 2 * new_range + new_min
        keep = np.abs(after) < 1.0 - 1e-6
        error = float(np.abs(raw_before - raw_after)[keep].max())
        print("  {}: {} windows, max raw disagreement {:.2e} m".format(split, len(array), error))
        if error > 1e-4:
            raise SystemExit("affine composition is wrong: {:.3e} m".format(error))

    if args.target_normalizer:
        shutil.copyfile(args.target_normalizer, staging / "normalizer.pt")
    else:
        torch.save({"data_min": torch.tensor(new_min, dtype=torch.float32),
                    "data_max": torch.tensor(new_max, dtype=torch.float32)},
                   staging / "normalizer.pt")
    normalizer_hash = sha256_file(staging / "normalizer.pt")

    build = json.loads((source / "build.json").read_text())
    build["artifacts"]["normalizer.pt"] = normalizer_hash
    build["representation_contract"]["normalization_artifact_sha256"] = normalizer_hash
    # The release contract wants the normalizer's own fit provenance, and the
    # honest answer here is "derived", not the original fit's report -- pointing
    # at that report would claim these numbers came out of that fit, which is
    # exactly the kind of quiet lie the contract exists to prevent.  A real
    # report is written beside the release and pointed at with its real hash.
    fit_report = staging / "normalizer_derivation.json"
    fit_report.write_text(json.dumps({
        "kind": "derived_percentile_refit" if not args.target_normalizer else "target_normalizer_units",
        "target_normalizer": (str(pathlib.Path(args.target_normalizer).resolve())
                              if args.target_normalizer else None),
        "target_normalizer_sha256": (sha256_file(args.target_normalizer)
                                     if args.target_normalizer else None),
        "derived_from_release": str(source.resolve()),
        "derived_from_normalizer_sha256": sha256_file(source / "normalizer.pt"),
        "method": ("every dimension re-expressed in the target normalizer's units by "
                   "inverting the source normalizer; nothing was fitted"
                   if args.target_normalizer else
                   "per-dimension percentile refit of the RAW values recovered by "
                   "inverting the source normalizer; all other dimensions are "
                   "copied unchanged from the source fit"),
        "percentiles": [args.low, args.high] if not args.target_normalizer else None,
        "dims": dims,
        "frames_examined": int(frames),
        "old_range_m": {str(d): float(old_range[d]) for d in dims},
        "new_range_m": {str(d): float(new_range[d]) for d in dims},
        "clipped_fraction_on_refit_dims": clipped_count / max(clipped_total, 1),
        "fit_split": "train",
    }, indent=2), encoding="utf-8")
    build["normalizer"] = dict(build["normalizer"])
    build["normalizer"].update({
        "copy_policy": ("byte_copy_of_target_normalizer_see_normalizer_derivation_json"
                        if args.target_normalizer else
                        "derived_percentile_refit_of_a_frozen_fit_see_normalizer_derivation_json"),
        "published_artifact_sha256": normalizer_hash,
        "source_artifact": str((output / "normalizer.pt").resolve()),
        "source_artifact_sha256": normalizer_hash,
        "fit_report": str((output / "normalizer_derivation.json").resolve()),
        "fit_report_sha256": sha256_file(fit_report),
    })
    for split in list(build["artifacts"]["splits"]):
        target_split = staging / split
        if not target_split.is_dir():
            continue
        for name in list(build["artifacts"]["splits"][split]):
            path = target_split / name
            if path.is_file():
                build["artifacts"]["splits"][split][name] = sha256_file(path)
    build["derived_from"] = {
        "release": str(source.resolve()),
        "tool": "tools/renormalize_release.py",
        "target_normalizer": args.target_normalizer,
        "percentiles": [args.low, args.high] if not args.target_normalizer else None,
        "dims": dims,
        "source_normalizer_sha256": sha256_file(source / "normalizer.pt"),
        "clipped_fraction_on_refit_dims": clipped_count / max(clipped_total, 1),
        "note": "affine recomposition of an already-normalized release; windows, "
                "labels, music and masks are the same bytes (hardlinked)",
    }
    (staging / "build.json").write_text(json.dumps(build, indent=2), encoding="utf-8")
    for name in ("windows.jsonl", "quarantine.jsonl"):
        if (source / name).is_file():
            os.link(source / name, staging / name)
    staging.rename(output)
    print("clipped {:.4f}% of the refit dimensions".format(100 * clipped_count / max(clipped_total, 1)))
    print(output)


if __name__ == "__main__":
    main()
