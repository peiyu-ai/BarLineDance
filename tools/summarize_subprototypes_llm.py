#!/usr/bin/env python3
"""Paper M3, step 2: the summarizing LLM that forms sub-prototypes.

The paper splits M3 across two models, not one.  A *tagging* VLM describes each
segment (``tools/caption_segments_vlm.py``), and then:

    "we use a summarizing LLM to iteratively: (i) identify a subset of mutually
    similar captions as a sub-prototype, and (ii) distill it into a concise
    semantic tag.  We repeat this procedure until the number of ungrouped
    segments falls below a threshold."

That second model is what this tool runs.  It matters that it is a *model* and
not a distance function: k-means over caption embeddings will split a group into
exactly the number of pieces it was asked for, wherever the geometry happens to
fall, while the paper's loop stops when the remainder is small and lets each
prototype yield as many sub-prototypes as it actually contains -- which is why
the paper reports an average of 7.3 rather than a constant.

The loop here is the paper's loop:

* captions are grouped by ``(prototype, genre)`` -- the genre pre-split;
* within a group the LLM is shown the *distinct* captions with how many
  segments each covers, and asked for one mutually-similar subset plus a tag;
* the chosen captions leave the pool, and the loop repeats until the ungrouped
  segments fall below ``--residual-threshold`` of the group or the LLM stops
  returning usable subsets;
* whatever is still ungrouped at that point is attached to the sub-prototype it
  agrees with most on the caption schema.  The paper does not say what becomes
  of its residue; assigning it by field agreement keeps every segment labelled,
  is deterministic, and is counted separately in the report so the LLM's share
  of the grouping is never overstated.

Deviation, recorded rather than hidden: the paper's summarizing LLM is
unnamed and its tagging VLM is Gemini-2.5-Pro; both are unreachable here, so a
local Qwen runs in their place.  The model name travels in the output header
and into every label row downstream.

Usage:
    python3 tools/summarize_subprototypes_llm.py \
        --captions runs/wild_captions_v1/captions.jsonl \
        --model third_party/QwenVL/Qwen3-VL-30B-A3B-Instruct/main \
        --output runs/wild_captions_v1/subprototypes.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.caption_segments_vlm import FIELDS, dtype_kwarg, model_name  # noqa: E402

SCHEMA_VERSION = "atomicdance-subprototype-summary-v1"
PRODUCER_VERSION = "qwen-summarizing-llm-v1"

PROMPT = (
    "Below are descriptions of short dance movements that a clustering step "
    "already placed in one group. Each line is one distinct description and "
    "the number of segments it covers.\n\n"
    "{items}\n\n"
    "Pick the ONE subset of lines that describe mutually similar movements -- "
    "movements a choreographer would call the same move. Prefer a subset of 2 "
    "to {cap} lines. Then name that subset with a short tag of at most six "
    "words.\n"
    'Reply with JSON only: {{"members": [<line numbers>], "tag": "<short tag>"}}'
)

# Greedy decoding makes a retry pointless unless the prompt changes, so the
# second attempt restates the requirement the first reply missed.
RETRY = ("\n\nYour previous reply was not usable. Reply with nothing but the JSON "
         "object, and put at least two line numbers in \"members\".")


class SummaryError(RuntimeError):
    pass


def group_key(row: Dict[str, object], genres: Dict[str, str]) -> Tuple[int, str]:
    """Prototype and genre -- the paper pre-splits each prototype by genre.

    The lookup is ``recluster_atomics_ingroup.genre_of``, not a dict ``get``.
    Recording ids come in two shapes across this repo -- the bundle's
    ``corpus:upload:clip`` and the cache's ``upload__clipNNN`` -- and the key is
    a property of the upload in both, so a plain ``get`` on the full id misses
    every row of the wild corpus and quietly returns "?".  Measured 2026-08-15:
    ``--genre-map`` was passed, the file loaded, and all 100 cells still came
    out flat, which is the pre-split disappearing from this stage while
    ``recluster_atomics_ingroup`` still applies it -- the two stages then
    disagree about what a group is, exactly as ``run_aist_m3_llm.sh`` warns.

    The reclusterer already resolves both shapes; sharing its function is what
    makes the two stages agree by construction rather than by coincidence.
    """
    from tools.recluster_atomics_ingroup import genre_of

    recording = str(row.get("recording_id", ""))
    return int(row["prototype"]), (genre_of(recording, genres) or "?" if genres
                                   else "?")


def agreement(a: Dict[str, str], b: Dict[str, str]) -> float:
    """Share of schema fields on which two captions agree."""
    if not a or not b:
        return 0.0
    # Over the fields actually present, not a module constant: schema v2 adds
    # rhythm and splits dynamics, and a hard-coded v1 tuple would score two v2
    # captions on six of their eight axes and silently ignore the three the
    # schema was extended for.
    fields = sorted(set(a) | set(b)) or list(FIELDS)
    return sum(1.0 for field in fields if a.get(field) == b.get(field)) / len(fields)


class DegenerateSubset(Exception):
    """The model claimed the whole offered pool is one movement.

    Recorded rather than silently accepted.  ``parse_reply`` had a floor of two
    members and no ceiling, and ``--subset-cap`` reached the model only as the
    word "Prefer" inside the prompt, so a reply of "all of these are the same
    move" ended a cell in one round with one sub-prototype.  Run over every
    cell, that makes the "w/ LLM" vocabulary *identical to the M2 x genre
    pre-split* while every health field in the report shows its best possible
    value: mean_subprototypes_per_group 1.0, stopped_because
    "residual_below_threshold", and segments_in_llm_formed_subprototypes at its
    arithmetic maximum -- a metric whose stated purpose is that the LLM's share
    is never overstated, peaking exactly when the LLM contributed nothing.

    It is not an adversarial reply either.  On the real re-keyed corpus the
    median cell holds three distinct captions, mean pairwise field agreement is
    0.472, and 33% of pairs agree on four of six fields; cell (2, gLH) is three
    captions differing only in `legs`.  "All three" is a defensible answer to
    "pick the ONE subset a choreographer would call the same move".
    """


def parse_reply(text: str, allowed: Sequence[int],
                subset_cap: Optional[int] = None) -> Optional[Tuple[List[int], str]]:
    """Read ``{"members": [...], "tag": "..."}`` out of a reply.

    Members outside the offered list are dropped rather than tolerated: a
    hallucinated line number would otherwise pull an unrelated caption into the
    sub-prototype, or -- worse, since ids are positions -- silently shift the
    membership of every later round.  A subset of one is rejected because a
    sub-prototype of a single caption is what the residual pass is for, and
    accepting them lets the loop "make progress" one caption at a time forever.
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
    permitted = set(int(value) for value in allowed)
    members: List[int] = []
    for value in raw.get("members") or []:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number in permitted and number not in members:
            members.append(number)
    if len(members) < 2:
        return None
    # Taking the whole pool is only degenerate when the pool is bigger than the
    # cap the prompt asked for.  A cell of two or three captions that really is
    # one movement should yield one sub-prototype -- that is a correct answer,
    # not a collapse.  But a model asked to prefer subsets of at most `cap` and
    # answering with all 40 has declined to group, and accepting that ends the
    # cell in one round.
    if subset_cap and len(members) >= len(permitted) > subset_cap:
        raise DegenerateSubset(
            "the reply selects all {} offered captions, more than the cap of {}".format(
                len(permitted), subset_cap))
    # The cap is enforced here, not merely suggested in the prompt.
    if subset_cap and len(members) > subset_cap:
        members = members[:subset_cap]
    tag = raw.get("tag")
    tag = tag.strip() if isinstance(tag, str) else ""
    return members, (tag or "untagged")


def render_items(pool: Sequence[Tuple[int, str, int]]) -> str:
    """Number the offered captions for the prompt."""
    return "\n".join("{}. {} [{} segments]".format(index, caption, count)
                     for index, caption, count in pool)


def merge_small(subprototypes: List[Dict[str, object]],
                entries: Sequence[Tuple[str, int, Dict[str, str]]], *,
                minimum: int) -> int:
    """Fold sub-prototypes below ``minimum`` into their most similar sibling.

    The paper's Fig. 4c puts only 15% of sub-prototypes under 20 samples.  A run
    that produces far more of them has split the prototype past the point where
    a class has enough data to be learned, whatever its average says -- and the
    average can look right while the median is a third of it.

    Merging is by caption-field agreement, the same rule the residue uses, and
    the count of merges is reported so the shape is never quietly manufactured.
    """
    if minimum <= 0 or len(subprototypes) < 2:
        return 0
    fields_of = {caption: fields for caption, _, fields in entries}

    def profile(sub):
        return [fields_of.get(caption, {}) for caption in sub["captions"]]

    merges = 0
    while len(subprototypes) > 1:
        smallest = min(range(len(subprototypes)),
                       key=lambda i: (subprototypes[i]["segments"], i))
        if subprototypes[smallest]["segments"] >= minimum:
            break
        source = subprototypes.pop(smallest)
        # The absorbed sub-prototype's residual_segments has to travel with it.
        # Dropping it re-credited field-assigned residue to the LLM, so the
        # reported LLM share rose as a function of how aggressively small
        # sub-prototypes were merged -- a headline number moving with a knob
        # that is supposed to only reshape, not re-attribute.
        carried_residual = int(source.get("residual_segments", 0) or 0)
        mine = profile(source)
        scores = [max((agreement(a, b) for a in mine for b in profile(other)), default=0.0)
                  for other in subprototypes]
        best = int(max(range(len(scores)), key=lambda i: (scores[i], -i)))
        subprototypes[best]["captions"].extend(source["captions"])
        subprototypes[best]["segments"] += source["segments"]
        subprototypes[best]["merged_segments"] = (
            subprototypes[best].get("merged_segments", 0) + source["segments"])
        if carried_residual:
            subprototypes[best]["residual_segments"] = (
                subprototypes[best].get("residual_segments", 0) + carried_residual)
        merges += 1
    return merges


def summarize_group(entries: Sequence[Tuple[str, int, Dict[str, str]]], ask: Callable[[str], str],
                    *, residual_threshold: float, max_rounds: int,
                    subset_cap: int, minimum_size: int = 0,
                    max_prompt_captions: Optional[int] = None) -> Dict[str, object]:
    """Run the paper's loop over one (prototype, genre) group.

    ``entries`` is one tuple per *distinct* caption: the sentence, how many
    segments carry it, and its schema fields.

    ``max_prompt_captions`` bounds how many of the remaining pool go into one
    prompt, and it is a memory bound rather than a modelling choice.  The
    weights of this 30B MoE hold 69.6 GiB of a 71.1 GiB card, so the forward has
    about 1 GiB to work in; ``run_wild_stage_g_m3a_oss`` already pins the
    captioner to batch 4 for the same reason, having measured the MoE asking for
    2.29 GiB at batch 8.  wild_v4's largest pre-split cell offers 225 distinct
    captions -- roughly 5,500 tokens -- and the forward asked for 2.04 GiB and
    died.

    The cap is not free and is not hidden: the pool is ordered by segment count,
    so a capped round shows the model the *most frequent* remaining captions and
    a rare one becomes visible only once the common ones have been grouped away.
    Rounds that were capped are counted and travel in the report, because a run
    where most rounds were capped saw a different problem than the paper's.
    """
    total = sum(count for _, count, _ in entries)
    remaining = {index: entry for index, entry in enumerate(entries)}
    subprototypes: List[Dict[str, object]] = []
    rounds = 0
    retries = 0
    capped_rounds = 0
    collapsed_first_round = False
    stopped = "residual_below_threshold"

    while remaining:
        ungrouped = sum(count for _, count, _ in remaining.values())
        if ungrouped <= residual_threshold * total:
            break
        if len(remaining) == 1:
            stopped = "single_caption_left"
            break
        if rounds >= max_rounds:
            stopped = "max_rounds"
            break
        rounds += 1
        pool = [(index, entry[0], entry[1]) for index, entry in sorted(remaining.items())]
        if max_prompt_captions and len(pool) > max_prompt_captions:
            # Ordered by segment count, not by index, so the cap keeps the
            # captions that cover the most segments rather than the ones that
            # happen to sort first.
            pool = sorted(pool, key=lambda item: -item[2])[:max_prompt_captions]
            pool.sort()
            capped_rounds += 1
        allowed = [index for index, _, _ in pool]
        prompt = PROMPT.format(items=render_items(pool), cap=subset_cap)
        degenerate = 0
        try:
            parsed = parse_reply(ask(prompt), allowed, subset_cap)
        except DegenerateSubset:
            parsed, degenerate = None, degenerate + 1
        if parsed is None:
            # One unusable reply should not end a whole prototype: greedy
            # decoding means the retry has to differ, so it is asked again with
            # the requirement spelled out.  Only a second failure stops.
            retries += 1
            try:
                parsed = parse_reply(ask(prompt + RETRY), allowed, subset_cap)
            except DegenerateSubset:
                parsed, degenerate = None, degenerate + 1
        if parsed is None:
            stopped = ("llm_called_the_whole_cell_one_movement" if degenerate
                       else "llm_returned_no_usable_subset")
            # Captured here: by the time the report is built the residual pass
            # has appended a sub-prototype, so `not subprototypes` no longer
            # distinguishes a first-round collapse from a normal ending.
            collapsed_first_round = bool(degenerate) and not subprototypes
            break
        members, tag = parsed
        subprototypes.append({
            "tag": tag,
            "captions": [remaining[index][0] for index in members],
            "segments": sum(remaining[index][1] for index in members),
            "source": "llm",
        })
        for index in members:
            del remaining[index]

    # Whatever is left joins the sub-prototype it most agrees with.  With no
    # sub-prototype yet -- an LLM that refused from the first round -- the group
    # stays whole, which is the honest outcome: one sub-prototype, not a
    # scattering invented by this function.
    residual_segments = sum(count for _, count, _ in remaining.values())
    if remaining and not subprototypes:
        subprototypes.append({
            "tag": "ungrouped",
            "captions": [entry[0] for entry in remaining.values()],
            "segments": residual_segments,
            "source": "residual",
        })
    elif remaining:
        fields_of = {index: entry[2] for index, entry in remaining.items()}
        representative = [
            [entry[2] for entry in entries if entry[0] in set(sub["captions"])]
            for sub in subprototypes
        ]
        for index, entry in sorted(remaining.items()):
            scores = [max((agreement(fields_of[index], other) for other in group), default=0.0)
                      for group in representative]
            best = int(max(range(len(scores)), key=lambda i: (scores[i], -i)))
            subprototypes[best]["captions"].append(entry[0])
            subprototypes[best]["segments"] += entry[1]
            subprototypes[best]["residual_segments"] = (
                subprototypes[best].get("residual_segments", 0) + entry[1])

    merged = merge_small(subprototypes, entries, minimum=minimum_size)

    return {
        "subprototypes": subprototypes,
        "rounds": rounds,
        "retries": retries,
        "capped_rounds": capped_rounds,
        # Only a *first-round* whole-pool reply is the degenerate case.  Ending
        # that way after several subsets have been formed is the normal terminal
        # condition -- what is left really is one movement by then -- and
        # counting it would make a healthy run look collapsed.
        "llm_called_the_whole_cell_one_movement": collapsed_first_round,
        "merged_small_subprototypes": merged,
        "stopped_because": stopped,
        "segments": total,
        "distinct_captions": len(entries),
        "residual_segments": residual_segments,
    }


class Summarizer:
    """A local chat model, decoded greedily so a rerun reproduces the run."""

    def __init__(self, model_dir: pathlib.Path, *, device: str, max_new_tokens: int):
        import torch
        import transformers
        from transformers import AutoConfig, AutoProcessor

        config = AutoConfig.from_pretrained(str(model_dir))
        architecture = (config.architectures or [""])[0]
        if not hasattr(transformers, architecture):
            raise SummaryError(
                "transformers {} has no {}; the installed version cannot load this "
                "model".format(transformers.__version__, architecture))
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(str(model_dir))
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.model = getattr(transformers, architecture).from_pretrained(
            str(model_dir), device_map=device,
            **{dtype_kwarg(transformers.__version__): torch.bfloat16})
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None)
        width = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(generated[0][width:], skip_special_tokens=True)


def load_rows(path: pathlib.Path) -> List[Dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_checkpoint(path: Optional[pathlib.Path]) -> Dict[Tuple[int, str], Dict[str, object]]:
    """Groups a previous run finished, keyed the way ``run`` keys them."""
    done: Dict[Tuple[int, str], Dict[str, object]] = {}
    if path is None or not path.is_file():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A row half-written when the process died.  Dropping it costs
                # one group's LLM calls; trusting it would put a truncated
                # grouping into the vocabulary.
                continue
            done[(int(row["prototype"]), str(row["genre"]))] = row
    return done


def run(*, captions: pathlib.Path, output: pathlib.Path, ask: Callable[[str], str],
        model_name: str, genres: Optional[Dict[str, str]] = None,
        residual_threshold: float = 0.03, max_rounds: int = 20,
        subset_cap: int = 6, minimum_size: int = 0,
        limit_groups: Optional[int] = None, shard: int = 0, num_shards: int = 1,
        checkpoint: Optional[pathlib.Path] = None,
        max_prompt_captions: Optional[int] = None,
        unresolved_tolerance: float = 0.05) -> Dict[str, object]:
    rows = load_rows(captions)
    if not rows:
        raise SummaryError("{} has no caption rows".format(captions))
    genres = genres or {}

    grouped: Dict[Tuple[int, str], List[Dict[str, object]]] = collections.defaultdict(list)
    for row in rows:
        grouped[group_key(row, genres)].append(row)

    if genres:
        unresolved = sum(len(members) for key, members in grouped.items() if key[1] == "?")
        share = unresolved / max(1, len(rows))
        print("pre-split: {} group(s), {:.1%} of rows unresolved".format(
            len(set(key[1] for key in grouped)), share), flush=True)
        if share > unresolved_tolerance:
            raise SummaryError(
                "a genre map was given but {:.1%} of rows did not resolve to a "
                "group, above the {:.0%} tolerance.  Those rows form one flat "
                "cell, so this stage would run without the pre-split while "
                "recluster_atomics_ingroup still applies it".format(
                    share, unresolved_tolerance))

    keys = sorted(grouped)
    if limit_groups is not None:
        keys = keys[:limit_groups]
    # Sharded over *cells*, which are independent by construction -- the paper's
    # loop never looks outside one (prototype, genre) group.  Without this the
    # wild corpus is one process for as long as it takes, and this tool wrote
    # its output only after the last group returned, so an interruption at cell
    # 99 of 100 lost every LLM call.  ``run_aist_m3_llm.sh`` says "the captioner
    # and the summarizer are both resumable" and then substantiates only the
    # captioner; on AIST the run happened to survive, so nobody found out.
    if num_shards > 1:
        keys = keys[shard::num_shards]

    done = load_checkpoint(checkpoint)
    if done:
        print("resume: {} group(s) already summarised".format(len(done)), flush=True)

    started = time.time()
    groups_out = []
    sink = checkpoint.open("a", encoding="utf-8") if checkpoint is not None else None
    for position, key in enumerate(keys):
        finished = done.get((int(key[0]), str(key[1])))
        if finished is not None:
            groups_out.append(finished)
            continue
        members = grouped[key]
        counter = collections.Counter(str(row["caption"]) for row in members)
        fields_for = {}
        for row in members:
            fields_for.setdefault(str(row["caption"]), dict(row.get("fields") or {}))
        entries = [(caption, count, fields_for[caption])
                   for caption, count in counter.most_common()]
        result = summarize_group(entries, ask, residual_threshold=residual_threshold,
                                 max_rounds=max_rounds, subset_cap=subset_cap,
                                 minimum_size=minimum_size,
                                 max_prompt_captions=max_prompt_captions)
        result["prototype"], result["genre"] = int(key[0]), key[1]
        groups_out.append(result)
        if sink is not None:
            # Appended and flushed as each cell finishes, so the unit of loss is
            # one cell rather than the whole run.
            sink.write(json.dumps(result, sort_keys=True) + "\n")
            sink.flush()
        print("  group {}/{} prototype={} genre={} -> {} sub-prototypes "
              "({} rounds, {})".format(position + 1, len(keys), key[0], key[1],
                                       len(result["subprototypes"]), result["rounds"],
                                       result["stopped_because"]), flush=True)
    if sink is not None:
        sink.close()

    counts = [len(group["subprototypes"]) for group in groups_out]
    sizes = [sub["segments"] for group in groups_out for sub in group["subprototypes"]]
    # A segment counts as LLM-grouped only if the LLM itself put it there.  The
    # residue attached afterwards by field agreement rides in the same
    # sub-prototypes, so it has to be subtracted rather than inherited.
    llm_segments = sum(sub["segments"] - sub.get("residual_segments", 0)
                       for group in groups_out for sub in group["subprototypes"]
                       if sub["source"] == "llm")
    report = {
        "schema_version": SCHEMA_VERSION,
        "producer_version": PRODUCER_VERSION,
        "model": model_name,
        "substitutes_for": "the paper's summarizing LLM (unnamed; its tagging VLM is "
                           "Gemini-2.5-Pro)",
        "captions": str(captions.resolve()),
        "genre_presplit": bool(genres),
        "residual_threshold": residual_threshold,
        "max_rounds": max_rounds,
        "minimum_subprototype_size": minimum_size,
        "merged_small_subprototypes": sum(g["merged_small_subprototypes"]
                                          for g in groups_out),
        "max_prompt_captions": max_prompt_captions,
        # How much of this run saw a truncated pool.  A run where most
        # rounds were capped was shown the most frequent captions and not
        # the cell, which is a different problem from the paper's.
        "capped_rounds": sum(g.get("capped_rounds", 0) for g in groups_out),
        "rounds_total": sum(g["rounds"] for g in groups_out),
        "groups": len(groups_out),
        "total_subprototypes": sum(counts),
        "mean_subprototypes_per_group": round(sum(counts) / max(1, len(counts)), 2),
        "mean_segments_per_subprototype": round(sum(sizes) / max(1, len(sizes)), 2),
        "segments_in_llm_formed_subprototypes": llm_segments,
        "segments_total": sum(sizes),
        "residual_segments": sum(group["residual_segments"] for group in groups_out),
        "paper_reference": {"subprototypes_per_prototype": 7.3, "samples_each": 31.8},
        # The criterion this file most needed and did not have.  Every other
        # health field peaks when the LLM does nothing: if it answers "all of
        # these are one movement" for every cell, each cell ends in one round
        # with one sub-prototype, mean_subprototypes_per_group is 1.0,
        # stopped_because is the healthy "residual_below_threshold", and
        # segments_in_llm_formed_subprototypes hits its arithmetic maximum --
        # the metric whose stated purpose is that the LLM's share is never
        # overstated, maximal exactly when the share is zero.  The vocabulary is
        # then the M2 x genre pre-split wearing an LLM's name, and it would pass
        # the music gate more easily than a real one, because genre tracks the
        # backing song.  So the degenerate outcome is named and counted here.
        "cells_the_llm_called_one_movement": sum(
            1 for g in groups_out if g.get("llm_called_the_whole_cell_one_movement")),
        "cells_yielding_one_subprototype": sum(1 for c in counts if c <= 1),
        "grouping_is_degenerate": bool(counts) and sum(counts) <= len(counts),
        "degenerate_means": ("every cell yielded at most one sub-prototype, so this "
                             "vocabulary is the pre-split rather than an LLM grouping; "
                             "do not report it as Tab. 2's w/ LLM row"),
        "elapsed_s": round(time.time() - started, 1),
        "groups_detail": groups_out,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captions", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--genre-map", type=pathlib.Path, default=None,
                        help="JSON {recording_id: genre} for the paper's genre pre-split")
    # The paper gives no value for its stopping threshold, only the shape it
    # produces: 7.3 sub-prototypes per prototype at 31.8 samples each.  These
    # two defaults were fitted to that shape on this corpus -- at 0.1/12 the
    # loop stopped after 3.0 sub-prototypes per group at 59 samples each, and
    # at 0.03/6 it lands on 8.25 at ~33.  Same calibration logic as Alg. 1's
    # L_min: a free parameter set from the one number the paper does report.
    parser.add_argument("--residual-threshold", type=float, default=0.03,
                        help="stop a group once this share of its segments is ungrouped")
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--subset-cap", type=int, default=6,
                        help="upper bound suggested to the LLM for one subset")
    parser.add_argument("--min-subprototype-size", type=int, default=0,
                        help="fold sub-prototypes smaller than this into their most "
                             "similar sibling; 0 leaves the LLM's split alone")
    parser.add_argument("--limit-groups", type=int, default=None,
                        help="only the first N groups; for prompt shakedown")
    parser.add_argument("--max-prompt-captions", type=int, default=None,
                        help="most captions shown in one prompt.  A memory "
                             "bound: this MoE leaves ~1 GiB over its weights "
                             "and a 225-caption pool asked the forward for "
                             "2.04 GiB.  Capped rounds are counted in the report")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1,
                        help="cells are independent, so a corpus that needs "
                             "hours can be split across cards")
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None,
                        help="append each finished cell here and skip it on a "
                             "re-run.  Without it the report is written only "
                             "after the last cell, so an interruption loses "
                             "every LLM call the run made")
    parser.add_argument("--merge-only", action="store_true",
                        help="assemble the report from --checkpoint without "
                             "loading the model; every cell must already be "
                             "there, and it says so if one is not")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    genres = json.loads(args.genre_map.read_text()) if args.genre_map else None
    def refuse(_prompt: str) -> str:
        raise SummaryError(
            "--merge-only, but a cell has no checkpointed result; the shards "
            "did not all finish, and assembling now would publish a vocabulary "
            "missing whole prototypes without saying so")

    try:
        ask = refuse if args.merge_only else Summarizer(
            args.model, device=args.device, max_new_tokens=args.max_new_tokens)
        report = run(captions=args.captions, output=args.output, ask=ask,
                     model_name=model_name(args.model), genres=genres,
                     residual_threshold=args.residual_threshold,
                     max_rounds=args.max_rounds, subset_cap=args.subset_cap,
                     minimum_size=args.min_subprototype_size,
                     limit_groups=args.limit_groups, shard=args.shard,
                     num_shards=args.num_shards, checkpoint=args.checkpoint,
                     max_prompt_captions=args.max_prompt_captions)
    except (SummaryError, FileNotFoundError) as error:
        raise SystemExit("error: {}".format(error))
    print(json.dumps({k: v for k, v in report.items() if k != "groups_detail"},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
