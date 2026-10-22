"""Bound remote retrieval work even after the caller's deadline expires."""

from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore
from contextvars import copy_context

_slots = BoundedSemaphore(8)
_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="fa-remote-retrieval")


class RetrievalBusy(RuntimeError):
    pass


def submit(function, *args):
    slots = _slots
    if not slots.acquire(blocking=False):
        raise RetrievalBusy("Remote retrieval capacity is occupied; use local evidence")
    try:
        future = _pool.submit(copy_context().run, function, *args)
    except BaseException:
        slots.release()
        raise
    future.add_done_callback(lambda _: slots.release())
    return future
