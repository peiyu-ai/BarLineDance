"""Stamp a visual-vocabulary release with the materialized-build contract.

``cluster_visual_atomics.py`` produces the split arrays but not the
``atomic-window-materialization-v1`` manifest that ``train_atomic.py`` requires
before it will load a release.  That contract is what stops an ad-hoc
directory of arrays from silently entering training, so the answer is to
*satisfy* it, not to bypass it.

The visual release is a label-swap over a reference release that already
carries a valid build: motion, music, normalizer and split structure are the
reference's own bytes (minus windows dropped for missing video features).
This tool therefore:

* rewrites ``windows.jsonl`` from the reference rows -- kept rows get their new
  ``array_index`` and the visual label provenance (label space id, producer
  hash, new per-split label file hashes); dropped rows move to
  ``quarantine.jsonl`` with an explicit reason;
* recomputes every artifact hash over the actual published bytes;
* carries the reference's normalizer/input-manifest provenance forward
  unchanged (same bytes, same fit), and swaps only the label manifest to the
  segmentation report that produced these labels.

The result passes ``validate_training_data_root`` on its own merits: every
hash is of real bytes in the new directory, and the provenance chain states
exactly where each part came from.
"""

import argparse
import hashlib
import json
from pathlib import Path

LABEL_SPACE = "visual_s3d_kmeans100_v1"
PRODUCER_VERSION = "visual-atomic-discovery-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True,
                        help="visual release dir from cluster_visual_atomics.py")
    parser.add_argument("--reference", type=Path, required=True,
                        help="reference release whose build.json is the template")
    args = parser.parse_args()

    release, reference = args.release, args.reference
    build_info = json.loads((release / "build.json").read_text(encoding="utf-8"))
    if build_info.get("status") != "candidate_pending_predictability_gate":
        raise SystemExit("release {} does not look like a visual candidate".format(release))
    reference_build = json.loads((reference / "build.json").read_text(encoding="utf-8"))

    segmentation_path = Path(build_info["segmentation"])
    producer_hash = sha256_file(segmentation_path)

    # Kept windows per split, in materialized order.
    kept = {}
    for split in ("train", "val", "test"):
        names = json.loads((release / split / "names.json").read_text(encoding="utf-8"))
        kept[split] = {name: index for index, name in enumerate(names)}

    label_hashes = {
        split: sha256_file(release / split / "labels.npy") for split in kept
    }
    mask_hashes = {
        split: sha256_file(release / split / "label_valid_mask.npy") for split in kept
    }

    windows_rows = []
    quarantine_rows = []
    with open(reference / "windows.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            split = row["split"]
            name = row["sample_name"]
            if name in kept.get(split, {}):
                row["array_index"] = kept[split][name]
                row["label_space_id"] = LABEL_SPACE
                row["producer_version"] = PRODUCER_VERSION
                row["producer_artifact_sha256"] = producer_hash
                row["label_labels_sha256"] = label_hashes[split]
                row["label_valid_mask_sha256"] = mask_hashes[split]
                windows_rows.append(row)
            else:
                quarantine_rows.append(
                    {
                        "sample_name": name,
                        "split": split,
                        "reason": "no visual features for the backing video "
                                  "at materialization time",
                        "window_id": row.get("window_id"),
                    }
                )

    with open(release / "windows.jsonl", "w", encoding="utf-8") as handle:
        for row in windows_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    # The reference's own quarantine stays relevant (it explains windows that
    # never reached materialization); append this release's additional drops.
    with open(release / "quarantine.jsonl", "w", encoding="utf-8") as handle:
        with open(reference / "quarantine.jsonl", encoding="utf-8") as source:
            for line in source:
                handle.write(line)
        for row in quarantine_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    split_artifacts = {}
    for split in kept:
        split_artifacts[split] = {
            name: sha256_file(release / split / name)
            for name in (
                "motion.npy", "music.npy", "labels.npy",
                "label_valid_mask.npy", "names.json", "retrieval_groups.json",
            )
        }

    build = dict(reference_build)
    build["artifacts"] = {
        "normalizer.pt": sha256_file(release / "normalizer.pt"),
        "splits": split_artifacts,
        "windows.jsonl": sha256_file(release / "windows.jsonl"),
        "quarantine.jsonl": sha256_file(release / "quarantine.jsonl"),
    }
    build["counts"] = dict(reference_build["counts"])
    build["counts"]["materialized_windows"] = {
        split: len(kept[split]) for split in kept
    }
    build["counts"]["quarantined_records"] = (
        reference_build["counts"].get("quarantined_records", 0) + len(quarantine_rows)
    )
    build["input_manifests"] = dict(reference_build["input_manifests"])
    build["input_manifests"]["labels.jsonl"] = {
        "path": str(segmentation_path.resolve()),
        "sha256": producer_hash,
    }
    build["materializer_version"] = "visual-release-finalizer-v1"
    build["label_space_id"] = LABEL_SPACE
    build["visual_candidate_build"] = build_info
    (release / "build.json").write_text(
        json.dumps(build, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("windows kept {}, quarantined (new) {}".format(
        len(windows_rows), len(quarantine_rows)))
    print("build.json stamped for {}".format(release))

    # Prove the point: the training loader must accept it now.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train_atomic import validate_training_data_root

    provenance = validate_training_data_root(str(release))
    print("contract validated: {}".format(provenance["release_contract_validated"]))


if __name__ == "__main__":
    main()
