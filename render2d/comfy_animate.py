"""Drive the repository's existing ComfyUI Wan-Animate workflow, one clip at a time.

WHY THIS AND NOT THE HAND-BUILT DIFFUSERS PATH.  A complete, already-working
ComfyUI lives at ``../ComfyUI_Wan``: the Wan2.2-Animate weights in the layout
they were published in, ``ComfyUI-WanVideoWrapper`` and ``ComfyUI-KJNodes``
installed, and a workflow
(``user/default/workflows/wanvideo_WanAnimate_example_01.json``) that already
wires up the parts a hand-rolled loader has to reinvent -- pose embeds, the face
mask from pose keypoints, segment-wise sampling with overlap, VAE tiling.
Reconstructing that in diffusers meant writing two key-name converters and
pinning two configs by tensor shape; it got the weights loading, but this is the
path the assets were staged for.

THE FRAME RATE IS THE ONE TRAP HERE.  The workflow loads and writes at **16
fps** while everything in AtomicDance is **30 fps**: left alone, a 17.5 s dance
comes out as a 33 s one and every beat lands late.  ``force_rate`` on the video
loader and ``frame_rate`` on the combiner are both set to the pose video's own
rate, and the result's duration is checked against the source afterwards.
"""
import argparse
import json
import os
import pathlib
import subprocess
import time
import urllib.request

# Root of the sibling checkouts (ComfyUI_Wan, Lodge, this repo); set E2E_ROOT.
E2E_ROOT = os.environ.get("E2E_ROOT", "/workspace/e2e")

# Overridable so several ComfyUI servers (one per GPU) can each drive their own clips.
SERVER = os.environ.get("COMFY_SERVER", "http://127.0.0.1:8188")
WORKFLOW = pathlib.Path(
    E2E_ROOT + "/ComfyUI_Wan/user/default/workflows/"
    "wanvideo_WanAnimate_example_01.json")
COMFY_ROOT = pathlib.Path(E2E_ROOT + "/ComfyUI_Wan")

NODE_IMAGE = 57         # LoadImage -- the character still
NODE_VIDEO = 63         # VHS_LoadVideo -- the driving pose video
NODE_EMBEDS = 62        # WanVideoAnimateEmbeds -- width, height, frames
NODE_SAVE = 30          # VHS_VideoCombine -- the animated result


SCHEMA = {}


def load_schema():
    """Per node class, the declared input order -- required first, then optional.

    A saved UI graph stores widget VALUES as a bare list, so turning it back
    into named inputs needs the order the node declares, and only the server
    knows it.
    """
    with urllib.request.urlopen(SERVER + "/object_info", timeout=180) as response:
        info = json.loads(response.read())
    primitives = {"INT", "FLOAT", "STRING", "BOOLEAN"}
    for name, entry in info.items():
        spec = entry.get("input", {})
        widgets = []
        for group in ("required", "optional"):
            for key, declaration in spec.get(group, {}).items():
                kind = declaration[0] if isinstance(declaration, list) else declaration
                if isinstance(kind, list) or kind in primitives:
                    widgets.append(key)
        SCHEMA[name] = widgets
    return info


def api_format(graph):
    """The UI graph as the /prompt endpoint wants it: {id: {class_type, inputs}}."""
    links = {}
    for link in graph.get("links", []):
        # [id, from_node, from_slot, to_node, to_slot, type]
        links[link[0]] = (link[1], link[2])
    # SetNode/GetNode are KJNodes' named-wire sugar: they exist only in the UI
    # graph and the /prompt endpoint rejects them as unknown node types.  They
    # are resolved here into the direct connection they stand for -- a GetNode
    # named "pose" is replaced by whatever a SetNode of that name was fed.
    setters = {}
    aliases = {}
    for node in graph["nodes"]:
        if node.get("type") == "Reroute":
            # A Reroute is a WIRE drawn as a node: one input, one output, no
            # behaviour.  api_format drops it, so a link THROUGH it resolves to
            # a node that is not in the prompt and the dangling-reference sweep
            # deletes the input -- silently, exactly as the SetNode case below
            # did.  In Kijai's MTV-Crafter example that cost
            # WanVideoImageToVideoEncode its ``start_image``: the character
            # still never reached the model, and the render came back as a
            # different person entirely, generated from the text prompt alone.
            for slot in node.get("inputs", []) or []:
                link = slot.get("link")
                if link is not None and link in links:
                    aliases[node["id"]] = links[link]
        if node.get("type") == "SetNode":
            name = (node.get("widgets_values") or [None])[0]
            for slot in node.get("inputs", []) or []:
                link = slot.get("link")
                if link is not None and link in links:
                    setters[name] = links[link]
                    # A SetNode is a PASSTHROUGH as well as a publisher: it has
                    # its own output socket and graphs wire straight out of it,
                    # alongside the GetNodes that fetch it by name.  Aliasing
                    # only the GetNodes leaves those direct links pointing at a
                    # node that api_format drops, and the dangling-reference
                    # sweep then DELETES the input silently.  In the
                    # SteadyDancer graph that cost WanVideoEncode its ``image``
                    # -- the pose pictures, i.e. the entire pose conditioning --
                    # and left WanVideoImageToVideoEncode on its stale widget
                    # defaults: 832x480 for a 480x832 video, and 81 frames for
                    # a 619-frame dance.  Nothing failed; it would simply have
                    # animated the character with no pose and a wrong shape.
                    aliases[node["id"]] = links[link]
    for node in graph["nodes"]:
        if node.get("type") == "GetNode":
            name = (node.get("widgets_values") or [None])[0]
            if name in setters:
                aliases[node["id"]] = setters[name]

    def resolve(source, index):
        """Follow Get/Set hops until a real node is reached."""
        seen = set()
        while source in aliases and source not in seen:
            seen.add(source)
            source, index = aliases[source]
        return source, index

    # A PrimitiveNode is a CONSTANT drawn as a node: it has no inputs and its
    # widget holds the value.  api_format drops the node, so a socket fed by one
    # has to become the literal, or the dangling-reference sweep deletes the
    # input -- which is how WanVideoSamplerSettings lost both ``cfg`` and
    # ``seed`` and was rejected for missing required inputs.
    primitives = {}
    for node in graph["nodes"]:
        if node.get("type") == "PrimitiveNode":
            values = node.get("widgets_values") or []
            if values and not isinstance(values[0], (dict, list)):
                primitives[node["id"]] = values[0]

    prompt = {}
    for node in graph["nodes"]:
        if node.get("type") in ("SetNode", "GetNode", "Note", "MarkdownNote",
                                "Reroute", "PrimitiveNode"):
            continue
        if node.get("mode") in (2, 4):           # muted or bypassed
            continue
        inputs = {}
        wired = set()
        for slot in node.get("inputs", []) or []:
            link = slot.get("link")
            if link is not None and link in links:
                source, index = resolve(*links[link])
                inputs[slot["name"]] = (primitives[source] if source in primitives
                                        else [str(source), index])
                wired.add(slot["name"])
        values = node.get("widgets_values")
        if isinstance(values, dict):
            inputs.update({k: v for k, v in values.items()
                           if not isinstance(v, (dict, list))})
        elif isinstance(values, list):
            # WIDGET ORDER COMES FROM THE SERVER, not from the saved graph.
            # The graph's ``inputs`` list only contains widgets that were
            # converted to sockets, so zipping the saved values against it
            # shifts everything: WanVideoSampler came out with scheduler=True,
            # start_step='comfy' and riflex_freq_index='dpm++_sde'.
            # /object_info declares the true order for each class.
            # The names that are NOT sockets on this node, in the server's
            # declared order.  A saved graph lists every socket it has -- both
            # the wired ones and the widgets that were converted -- so the
            # widget values line up with "declared order minus this node's own
            # sockets".  Filtering by ``inputs`` instead (the ones that ended up
            # wired) leaves the converted-but-unwired sockets in the list and
            # shifts everything by one: ImageResizeKJv2 came out with
            # crop_position='pad_edge_pixel', which is keep_proportion's value.
            # Which names carry widget VALUES is decided by TYPE, not by what
            # the saved graph happens to have turned into a socket.  A widget is
            # an input whose declared type is a primitive (INT/FLOAT/STRING/
            # BOOLEAN) or a list of choices; IMAGE, MASK, VAE and friends never
            # are.  Two earlier rules both failed on ImageResizeKJv2: zipping
            # against the graph's socket list gave crop_position the value of
            # keep_proportion, and excluding every socket dropped width, height
            # and five more as "required input missing" because they had been
            # converted to sockets while still holding their values.
            # No extra filtering: SCHEMA already holds ONLY the widget names,
            # in declared order, so the saved values line up with it one to one.
            # Subtracting ``inputs`` here shifted the list whenever a wired
            # socket happened to share a name, which is how crop_position ended
            # up holding keep_proportion's value.
            order = SCHEMA.get(node["type"], ())
            # A ``seed``/``noise_seed`` widget is saved as TWO entries: the value
            # and its control mode ('fixed', 'increment', 'randomize'), which is
            # a UI affordance and not an input.  Dropping the extra keeps every
            # later value aligned -- with it in place WanVideoSampler received
            # scheduler=True, riflex_freq_index='dpm++_sde' and start_step='comfy'.
            values = list(values)
            for position, name in enumerate(order):
                if name in ("seed", "noise_seed") and position + 1 < len(values):
                    following = values[position + 1]
                    if following in ("fixed", "increment", "decrement", "randomize"):
                        del values[position + 1]
                    break
            for name, value in zip(order, values):
                if isinstance(value, (dict, list)):
                    continue
                # Some nodes store a rendered PREVIEW in their widget list (the
                # scheduler keeps a base64 plot of its sigma curve).  It is a UI
                # artefact, not an input, and passing it makes the prompt tens
                # of kilobytes of base64 per node.
                if isinstance(value, str) and value.startswith("<img src='data:"):
                    continue
                # A WIRE BEATS THE SAVED WIDGET VALUE.  A widget converted to a
                # socket keeps its last value in ``widgets_values`` -- the list
                # stays positionally complete, which is what makes the zip work
                # -- so assigning unconditionally overwrites the connection with
                # a stale number.  In the SteadyDancer graph that put
                # WanVideoImageToVideoEncode back on 832x480 and 81 frames for a
                # 480x832, 619-frame dance, silently, because all three of
                # width, height and num_frames are wired to the video's own
                # size.  The positional consumption must still happen, so this
                # skips the ASSIGNMENT, not the iteration.
                if name in wired:
                    continue
                inputs[name] = value
        prompt[str(node["id"])] = {"class_type": node["type"], "inputs": inputs}
    return prompt


def use_our_pose(graph):
    """Feed OUR pose video in where the workflow would extract one from footage.

    The published workflow starts from a real dance video and runs
    ``DWPreprocessor`` over it to get the OpenPose skeleton.  We already have
    that skeleton -- it is what ``project_pose_2d.py`` produced from the
    generated 3D dance -- so the preprocessor is not just unnecessary, it is not
    installed here and the graph is rejected without it.

    The rewire is small because the workflow passes everything by name: the
    ``SetNode`` that publishes ``pose_images`` is pointed at the video loader
    instead of at the preprocessor, and the face branch (which needs the
    preprocessor's KEYPOINTS, a second output we do not have) is dropped, with
    the character still standing in for ``face_images``.  Dropping it is honest
    for this demo: there is no face to track in a stick figure.
    """
    by_id = {n["id"]: n for n in graph["nodes"]}
    loader = next(n for n in graph["nodes"] if n["type"] == "VHS_LoadVideo")
    image = next(n for n in graph["nodes"] if n["type"] == "LoadImage")

    def named(kind, name):
        for node in graph["nodes"]:
            if node["type"] == kind and (node.get("widgets_values") or [None])[0] == name:
                return node
        return None

    # Re-point each publisher at a source we actually have.
    repoint = {"pose_images": loader["id"], "face_images": image["id"]}
    next_link = max((l[0] for l in graph["links"]), default=0) + 1
    keep = []
    for link in graph["links"]:
        keep.append(link)
    for name, source in repoint.items():
        setter = named("SetNode", name)
        if setter is None:
            continue
        slot = (setter.get("inputs") or [{}])[0]
        keep = [l for l in keep if l[0] != slot.get("link")]
        keep.append([next_link, source, 0, setter["id"], 0, "IMAGE"])
        slot["link"] = next_link
        next_link += 1
    graph["links"] = keep

    # Everything that exists to DERIVE inputs from real footage is removed, and
    # it is exactly the set this installation lacks -- asked directly,
    # /object_info knows 1091 node types and the workflow's only four unknowns
    # are DWPreprocessor, PixelPerfectResolution, Sam2Segmentation and
    # DownloadAndLoadSAM2Model.  Pose extraction we do ourselves; background
    # segmentation we do not want, the character image being the background.
    drop = {"DWPreprocessor", "PixelPerfectResolution", "Sam2Segmentation",
            "DownloadAndLoadSAM2Model", "FaceMaskFromPoseKeypoints",
            "ImageCropByMaskAndResize", "GrowMaskWithBlur", "InvertMask",
            # These consume the mask branch that went with the segmenter.
            "DrawMaskOnImage", "GrowMask", "MaskPreview", "BlockifyMask",
            "ImageCompositeMasked", "MaskToImage", "ImageToMask"}
    graph["nodes"] = [n for n in graph["nodes"] if n["type"] not in drop]

    # Any publisher left without a source now resolves to nothing, which is
    # what ``bg_images`` and ``mask`` should be: WanVideoAnimateEmbeds treats
    # both as optional.
    alive = {n["id"] for n in graph["nodes"]}
    graph["links"] = [l for l in graph["links"] if l[1] in alive and l[3] in alive]
    for node in graph["nodes"]:
        for slot in node.get("inputs", []) or []:
            if slot.get("link") is not None and not any(
                    l[0] == slot["link"] for l in graph["links"]):
                slot["link"] = None
    return graph


def fix_model_names(prompt):
    """Point every loader at a file the server actually lists.

    The published workflow carries Windows paths (``WanVideo\\2_2\\...``) and
    filenames from whoever exported it; this installation lists its own.  Rather
    than hard-coding a translation, each loader's declared choices are read back
    from /object_info and the closest match by basename is used -- so a weight
    that was staged under a slightly different name still resolves, and one that
    is genuinely absent fails with the list of what IS there.
    """
    with urllib.request.urlopen(SERVER + "/object_info", timeout=120) as response:
        info = json.loads(response.read())
    for node in prompt.values():
        spec = info.get(node["class_type"], {}).get("input", {}).get("required", {})
        for name, value in list(node["inputs"].items()):
            if not isinstance(value, str) or name not in spec:
                continue
            choices = spec[name][0]
            if not isinstance(choices, list) or value in choices:
                continue
            stem = value.replace("\\", "/").replace("\\\\", "/").split("/")[-1]
            match = next((c for c in choices if c.split("/")[-1] == stem), None)
            if match is None:
                # Fall back on the longest shared prefix of the basenames.  The
                # published workflow asks for Wan2_1_VAE while an Animate run
                # needs Wan2_2_VAE -- the names differ by one character and the
                # right answer is the one staged for this model, so a prefix
                # match picks it and the substitution is printed.
                def shared(candidate):
                    other = candidate.split("/")[-1]
                    n = 0
                    while n < min(len(stem), len(other)) and stem[n] == other[n]:
                        n += 1
                    return n
                match = max(choices, key=shared)
                # 4 characters, not 6: the pair this exists for is
                # ``Wan2_1_VAE_bf16`` against ``Wan2_2_VAE_bf16``, whose common
                # prefix is exactly ``Wan2`` -- a 6-character floor rejected the
                # very substitution it was written to make.
                if shared(match) < 4:
                    match = None
                else:
                    print("  {}: {!r} -> {!r}".format(node["class_type"], value, match))
            if match is None:
                raise SystemExit(
                    "node {} wants {}={!r} but the server offers {}"
                    .format(node["class_type"], name, value, choices))
            node["inputs"][name] = match
    return prompt


def post(prompt):
    # COMFY_FRONT=1 queues this job at the HEAD of the server's queue (ComfyUI's own "front" flag): the fast
    # iteration renders go ahead of hour-long batches that share the same servers.
    payload = {"prompt": prompt}
    if os.environ.get("COMFY_FRONT"):
        payload["front"] = True
    data = json.dumps(payload).encode()
    request = urllib.request.Request(SERVER + "/prompt", data=data,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        # The body carries which node and which input ComfyUI rejected; without
        # it a 400 says only "Bad Request" and the graph has 66 nodes.
        detail = error.read().decode("utf-8", "replace")
        raise SystemExit("ComfyUI rejected the prompt ({}):\n{}".format(
            error.code, detail[:3000]))


def wait(prompt_id, timeout=7200):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with urllib.request.urlopen(
                "{}/history/{}".format(SERVER, prompt_id), timeout=30) as response:
            history = json.loads(response.read())
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("completed"):
                return entry
            if status.get("status_str") == "error":
                raise SystemExit("ComfyUI reported an error:\n{}".format(
                    json.dumps(status, indent=2)[:2000]))
        time.sleep(5)
    raise SystemExit("timed out after {}s waiting for {}".format(timeout, prompt_id))


def probe(path):
    """{width, height, fps, frames} of a video, by decoding it.

    A DICT and not a tuple, because ffprobe's csv output is in the STREAM's
    field order (width, height, r_frame_rate, nb_read_frames) and not in the
    order the fields were asked for.  Positional unpacking therefore reads
    correctly at one call site and silently swaps at another: the duration
    check at the end of a render was comparing the video's WIDTH against a
    frame count, which turned a correct 329-frame / 20.6 s result into
    "480 frames at 16 fps = 30.0 s" and failed the clip.
    """
    fields = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip().split(",")
    rate = fields[2]
    return {"width": int(fields[0]), "height": int(fields[1]),
            "fps": (eval(rate) if "/" in rate else float(rate)),
            "frames": int(fields[3])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", required=True, help="path under ComfyUI input/")
    ap.add_argument("--pose-video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=832)
    args = ap.parse_args()

    info = probe(args.pose_video)
    width, height, fps, frames = (info["width"], info["height"],
                                  info["fps"], info["frames"])
    print("pose video: {}x{} {} frames at {:g} fps".format(width, height, frames, fps))

    # ComfyUI reads from its own input/ directory.
    staged = COMFY_ROOT / "input" / pathlib.Path(args.pose_video).name
    if staged.resolve() != pathlib.Path(args.pose_video).resolve():
        staged.write_bytes(pathlib.Path(args.pose_video).read_bytes())

    load_schema()
    graph = json.loads(WORKFLOW.read_text())
    graph = use_our_pose(graph)
    prompt = api_format(graph)
    prompt[str(NODE_IMAGE)]["inputs"]["image"] = pathlib.Path(args.character).name
    video = prompt[str(NODE_VIDEO)]["inputs"]
    video["video"] = staged.name
    video["force_rate"] = int(round(fps))        # NOT the workflow's 16
    video["custom_width"] = args.width
    video["custom_height"] = args.height
    video["frame_load_cap"] = 0
    embeds = prompt[str(NODE_EMBEDS)]["inputs"]
    embeds["width"] = args.width
    embeds["height"] = args.height
    embeds["num_frames"] = frames
    save = prompt[str(NODE_SAVE)]["inputs"]
    save["frame_rate"] = int(round(fps))
    save["save_output"] = True
    save["filename_prefix"] = "atomicdance_2d"

    # Only the node that writes the animation survives: the workflow has three
    # VHS_VideoCombine nodes, two of which preview the branches that were just
    # removed, and a graph is rejected if any OUTPUT node cannot be validated.
    for node_id, node in list(prompt.items()):
        if node["class_type"] == "VHS_VideoCombine" and int(node_id) != NODE_SAVE:
            del prompt[node_id]

    # Any input still pointing at a node that was removed is dropped: the
    # optional ones (bg_images, mask) are exactly what the segmentation branch
    # fed, and ComfyUI validates a dangling reference as KeyError '131' rather
    # than as "this optional input is absent".
    for node in prompt.values():
        for name, value in list(node["inputs"].items()):
            if isinstance(value, list) and len(value) == 2 and value[0] not in prompt:
                del node["inputs"][name]

    prompt = fix_model_names(prompt)
    result = post(prompt)
    print("queued", result.get("prompt_id"))
    entry = wait(result["prompt_id"])

    produced = []
    for node_output in entry.get("outputs", {}).values():
        for item in node_output.get("gifs", []) + node_output.get("videos", []):
            produced.append(COMFY_ROOT / item.get("subfolder", "") / item["filename"])
    produced = [p for p in produced if p.is_file()]
    if not produced:
        raise SystemExit("the workflow completed but wrote no video; outputs were "
                         + json.dumps(entry.get("outputs", {}))[:500])
    newest = max(produced, key=lambda p: p.stat().st_mtime)

    target = pathlib.Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.audio and pathlib.Path(args.audio).is_file():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(newest),
                        "-i", args.audio, "-c:v", "copy", "-c:a", "aac",
                        "-shortest", str(target)], check=True)
    else:
        target.write_bytes(newest.read_bytes())

    got_frames = probe(target)["frames"]
    print("{} -> {} ({} frames against the pose video's {})"
          .format(newest.name, target, got_frames, frames))
    if abs(got_frames - frames) > max(8, frames * 0.05):
        print("  WARNING: frame count differs by more than 5%; the dance will "
              "not line up with its music")


if __name__ == "__main__":
    main()
