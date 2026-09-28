"""The one thing in the extraction driver that can be tested without GVHMR.

``run_gvhmr_extract.py`` needs the vendored GVHMR checkout, its checkpoints and
a GPU, so nothing here can run it.  But the defect that cost the most wall
clock on this corpus was not in the model path at all -- it was a
multiprocessing sharing strategy, and that *is* testable, so it is pinned here.

Stage B lost 15 of 21 workers permanently to it.  GVHMR's SLAM reader is a
separate process that feeds frames over a ``multiprocessing.Queue``, and every
item carries a torch tensor.  Under torch's default ``file_descriptor``
strategy the storage travels as a file descriptor the consumer fetches over the
*producer's* ``resource_sharer`` socket; the reader exits when it runs out of
frames, the socket goes with it, and tensors still in the queue raise
``FileNotFoundError``.  The consumer then stops draining, the producer's feeder
thread blocks on a full pipe, and both deadlock in their atexit join.  Nothing
reports it: the process is still there, with no traceback and no progress.
"""

import multiprocessing
import pathlib
import runpy
import sys
import time

import pytest
import torch.multiprocessing as tmp

REPO = pathlib.Path(__file__).resolve().parents[1]
DRIVER = REPO / "tools" / "run_gvhmr_extract.py"


@pytest.fixture(autouse=True)
def restore_strategy():
    """The strategy is process-global; leaking it would change other tests."""
    before = tmp.get_sharing_strategy()
    yield
    tmp.set_sharing_strategy(before)


def test_import_sets_file_system_sharing():
    """Module scope, not ``main()``, is what reaches the spawned reader.

    DPVO calls ``set_start_method('spawn', True)`` at import, so the reader
    does not inherit this process's strategy.  A spawned child re-imports
    ``sys.argv[0]`` as ``__mp_main__``, which makes module scope the only place
    a setting lands on both sides -- moving this call inside ``main()`` would
    leave the child on the default and put the deadlock back.
    """
    tmp.set_sharing_strategy("file_descriptor")
    runpy.run_path(str(DRIVER), run_name="__mp_main__")
    assert tmp.get_sharing_strategy() == "file_system"


COUNT = 16
# Each item is what video_stream puts: a decoded frame and the intrinsics.
# The split matters.  The frame is a numpy array, pickled *by value*, so it is
# what fills the 64 KB pipe and leaves the feeder thread blocked -- which is
# what keeps the reader alive inside its atexit join rather than exiting
# outright.  The intrinsics are a torch tensor, pickled as a *handle* into
# shared memory, so they are the part that can go missing when the producer
# tears down.  Sending only tensors reproduces neither: the payloads are tiny,
# the feeder drains, the reader exits, and no strategy survives that.
FRAME_SHAPE = (256, 256, 2)  # 128 KB, against a 64 KB pipe buffer


def _produce(queue, strategy, returned):
    """The reader's side.  It sets the strategy itself because a *spawned*
    child does not inherit the parent's -- which is the whole reason the
    driver puts its call at module scope, where the child's re-import of
    ``__main__`` runs it.
    """
    import numpy as np
    import torch
    import torch.multiprocessing as child_tmp

    child_tmp.set_sharing_strategy(strategy)
    for i in range(COUNT):
        frame = np.full(FRAME_SHAPE, i % 256, dtype=np.uint8)
        queue.put((i, frame, torch.full((4,), float(i))))
    returned.set()
    # Returning here is what GVHMR's video_stream does once ``queue.empty()``
    # says the consumer has caught up -- and ``empty()`` is exactly the
    # unreliable judgement that lets it happen with frames still in flight.


def _drain_while_producer_shuts_down(strategy):
    """Model the window stage B actually fails in.

    The reader has finished its function and is inside interpreter shutdown --
    resource_sharer already stopped, feeder thread still blocked on a full pipe
    -- while the consumer is only now asking for the frames it never read.
    """
    context = multiprocessing.get_context("spawn")
    # Unbounded on purpose.  ``put`` then returns as soon as the item is on the
    # internal buffer, so the reader reaches the end of its function while the
    # feeder thread is still blocked writing to a full pipe -- which is the
    # state that matters.  A bounded queue would instead block the reader until
    # the consumer drains, and the consumer here is deliberately late.
    queue = context.Queue()
    returned = context.Event()
    reader = context.Process(target=_produce, args=(queue, strategy, returned))
    reader.start()
    try:
        assert returned.wait(timeout=120), "reader never finished producing"
        for _ in range(200):  # let it get into atexit, not merely past return
            if not reader.is_alive():
                break
            time.sleep(0.05)
        assert reader.is_alive(), "reader exited outright; not the production window"
        # The tensor, not the frame: the frame arrives by value either way.
        return [float(queue.get(timeout=60)[2][0]) for _ in range(COUNT)]
    finally:
        # Short on purpose.  In the file_descriptor case the reader is
        # deadlocked by construction and this timeout is always paid, so it is
        # the suite's floor -- and a reader that is going to exit has nothing
        # left to do but exit.
        reader.join(timeout=15)
        if reader.is_alive():
            reader.kill()
            reader.join(timeout=30)


@pytest.mark.skipif(sys.platform != "linux", reason="fd sharing is a POSIX path")
def test_file_system_survives_the_producer_shutting_down():
    """What the fix buys: shm outlives the producer's resource_sharer."""
    assert _drain_while_producer_shuts_down("file_system") == [
        float(i) for i in range(COUNT)]


@pytest.mark.skipif(sys.platform != "linux", reason="fd sharing is a POSIX path")
def test_file_descriptor_loses_them_and_is_why_stage_b_wedged():
    """And what it buys it *from* -- the exact error stage B printed 15 times.

    Pinned because the symptom is otherwise unrecognisable: ``[Errno 2] No
    such file or directory`` names no file, and the file it means is a
    ``/tmp`` socket that no longer exists.

    Retried, and skipped rather than failed when it will not reproduce.  This
    direction *is* a race -- it needs the reader to reach shutdown while items
    are outstanding, and in production it fired on 15 clips out of 4,750.  A
    machine under load can shift the timing enough that the drain wins.
    Asserting a race once is how a suite acquires a test that fails for reasons
    unrelated to the code, and this file already had to fix that on the other
    direction.
    """
    for _ in range(5):
        try:
            _drain_while_producer_shuts_down("file_descriptor")
        except FileNotFoundError:
            return
    pytest.skip("the shutdown race did not reproduce in five attempts; the "
                "mechanism is documented above and the file_system direction, "
                "which is the one the fix relies on, is asserted")
