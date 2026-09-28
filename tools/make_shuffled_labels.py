#!/usr/bin/env python3
"""A negative control for the prototype sheets: same groups, shuffled membership.

Why this exists.  A page of prototype cards looks coherent whether or not the
clustering found anything, because the reader is a pattern matcher and five
dancers in canonical pose always rhyme a little.  Read alone it is CLAUDE.md
section 2's gate that cannot fail.  So the real sheet is published next to a
sheet built the same way from labels that carry no information, and the question
becomes one that *can* come out either way: can you tell which page is which.

What is held fixed, and why each one matters:

* **Segment boundaries.**  Untouched.  Both arms show the same spans, so a
  difference cannot come from one arm having tidier cuts.
* **Group sizes.**  The label vector over the segments the renderers actually
  draw is *permuted*, not redrawn, so every prototype keeps its exact size.
  This matters because ``select_subprototypes`` walks the size-ordered list;
  redrawing uniformly would give the control 53 near-equal groups, and the two
  pages would then differ in which part of the size range they sample rather
  than in whether their members match.
* **Uploads per group.**  Not held fixed, and cannot be -- scattering
  membership necessarily raises upload diversity.  Both renderers draw members
  from distinct uploads regardless, so the confound that spread exists to
  control is controlled in both arms.

What is destroyed: only the association between a segment's motion and the
group it sits in.  That is the thing under test.

Two hazards, both turned into gates rather than warnings.

*The pool must be exactly the drawn set.*  ``collect_members`` keeps runs with
``label > 0 and end - start >= 4``.  Permuting over a wider pool would let a
label the real arm spends on a drawn segment land on an undrawn one, and the
sizes would drift.  So the pool is that predicate, verbatim.

*Touching runs fuse.*  ``segments_of`` recovers segments as runs of equal label,
so if the permutation gives the same label to two runs that abut, they merge
into one longer segment and the size table moves.  Conflicts are swapped apart,
and then -- because a repair loop that silently gave up would look identical --
the sizes are re-derived from the written tree and compared against the real
tree's.  A mismatch is a non-zero exit.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import shutil
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.recluster_atomics_ingroup import segments_of  # noqa: E402

MIN_FRAMES = 4          # collect_members' own floor; kept in lockstep with it


def load_rows(labels_dir: pathlib.Path):
    return [json.loads(line) for line
            in (labels_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]


def group_sizes(labels_dir: pathlib.Path) -> collections.Counter:
    """Exactly the grouping ``build_atomic_gallery.collect_members`` derives."""
    sizes: collections.Counter = collections.Counter()
    for row in load_rows(labels_dir):
        labels = np.load(labels_dir / row["labels_path"])
        for start, end, label in segments_of(labels):
            if label <= 0 or end - start < MIN_FRAMES:
                continue
            sizes[int(label)] += 1
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=pathlib.Path,
                        help="a materialised labels tree (labels.jsonl + arrays)")
    parser.add_argument("target", type=pathlib.Path,
                        help="where to write the size-matched shuffled tree")
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    source, target = args.source, args.target

    rows = load_rows(source)
    arrays, slots, dtypes = {}, [], {}
    # Every positive run, drawn or not.  A drawn run that fuses with an undrawn
    # short one grows, which moves the span the card renders; that is as much a
    # broken control as a size change, so both kinds enter conflict detection.
    positive_runs = collections.defaultdict(list)
    for index, row in enumerate(rows):
        original = np.load(source / row["labels_path"])
        dtypes[index] = original.dtype
        arrays[index] = np.array(original, dtype=np.int64)
        for start, end, label in segments_of(arrays[index]):
            if label <= 0:
                continue
            drawn = end - start >= MIN_FRAMES
            if drawn:
                slots.append((index, start, end))
            positive_runs[index].append([start, end, len(slots) - 1 if drawn else None,
                                         int(label)])

    pool = np.array([int(arrays[i][s]) for i, s, _ in slots], dtype=np.int64)
    abutting = sum(1 for runs in positive_runs.values()
                   for a, b in zip(runs, runs[1:]) if a[1] == b[0])
    print("{} recordings, {} drawn segments, {} distinct labels".format(
        len(rows), len(slots), len(set(pool.tolist()))), flush=True)
    print("{} places where two positive runs abut (fusion candidates)".format(
        abutting), flush=True)

    rng = np.random.default_rng(args.seed)
    assigned = pool[rng.permutation(len(pool))]

    def label_at(run) -> int:
        """A run's label after the permutation; undrawn runs never moved."""
        return int(assigned[run[2]]) if run[2] is not None else run[3]

    # To a fixed point, not one pass.  A swap that separates one abutting pair
    # can hand the other end of the swap a label that collides with a pair
    # already walked past -- which is exactly what the size gate caught on the
    # first version of this script (10,651 segments against 10,652).
    conflicts = [(a, b) for runs in positive_runs.values()
                 for a, b in zip(runs, runs[1:]) if a[1] == b[0]]
    swaps, sweeps = 0, 0
    while True:
        live = [(a, b) for a, b in conflicts if label_at(a) == label_at(b)]
        if not live:
            break
        sweeps += 1
        if sweeps > 100:
            raise SystemExit("repair did not converge in 100 sweeps")
        for a, b in live:
            if label_at(a) != label_at(b):
                continue                      # an earlier swap already fixed it
            movable = b if b[2] is not None else a
            if movable[2] is None:
                raise SystemExit("two undrawn runs abut with equal labels")
            guard = 0
            while label_at(a) == label_at(b):
                other = int(rng.integers(len(assigned)))
                assigned[movable[2]], assigned[other] = (
                    assigned[other], assigned[movable[2]])
                swaps, guard = swaps + 1, guard + 1
                if guard > 1000:
                    raise SystemExit("could not separate touching runs")
    print("{} repair swaps over {} sweeps".format(swaps, sweeps), flush=True)

    moved = int((assigned != pool).sum())
    for position, (index, start, end) in enumerate(slots):
        arrays[index][start:end] = assigned[position]

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    (target / "labels.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    for index, row in enumerate(rows):
        out = target / row["labels_path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, arrays[index].astype(dtypes[index]))
        mask = row.get("label_valid_mask_path")
        if mask and (source / mask).is_file():
            (target / mask).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / mask, target / mask)

    real, fake = group_sizes(source), group_sizes(target)
    print("real: {} groups, {} segments, sizes {}..{}".format(
        len(real), sum(real.values()), min(real.values()), max(real.values())), flush=True)
    print("ctrl: {} groups, {} segments, sizes {}..{}".format(
        len(fake), sum(fake.values()), min(fake.values()), max(fake.values())), flush=True)
    # The expected fixed fraction is NOT 1/K unless the groups are equal-sized:
    # under a permutation of the label multiset a segment keeps its label with
    # probability sum_i (n_i/N)^2.  Hard-coding 0.9 below made this gate
    # K-dependent -- it passed at K=53 (1.9% expected fixed) and fired
    # spuriously at K=5 (20% expected fixed), rejecting a perfectly good control.
    counts = np.array(sorted(real.values()), dtype=np.float64)
    expected_moved = 1.0 - float(((counts / counts.sum()) ** 2).sum())
    print("{} of {} segments changed label ({:.1%}; a permutation over these {} "
          "group sizes is expected to move {:.1%})".format(
              moved, len(pool), moved / len(pool), len(real), expected_moved),
          flush=True)

    ok = True
    if real != fake:
        differ = {k for k in set(real) | set(fake) if real[k] != fake[k]}
        print("FAIL: per-label sizes moved on {} labels -- the control is not "
              "size-matched, so a visible difference could be size not "
              "coherence".format(len(differ)), flush=True)
        ok = False
    if moved / len(pool) < 0.9 * expected_moved:
        print("FAIL: {:.1%} of segments moved against an expected {:.1%}; the "
              "permutation barely shuffled".format(
                  moved / len(pool), expected_moved), flush=True)
        ok = False
    print("OK: sizes identical label-for-label, membership scattered" if ok
          else "control REJECTED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
