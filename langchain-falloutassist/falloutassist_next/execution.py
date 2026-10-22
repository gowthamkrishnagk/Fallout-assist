"""Bounded read/model stages. Only the graph thread commits accepted results."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from threading import BoundedSemaphore
from time import monotonic

deadline_context = ContextVar("fa_operation_deadline", default=None)
_slots = BoundedSemaphore(8)
_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="fa-investigation")


class StageUnavailable(RuntimeError):
    pass


def remaining(default: float) -> float:
    deadline = deadline_context.get()
    value = min(default, deadline - monotonic()) if deadline is not None else default
    if value <= 0:
        raise TimeoutError("Investigation stage deadline expired")
    return value


def call(function, seconds: float, *args, **kwargs):
    if seconds <= 0:
        raise TimeoutError("Investigation stage deadline expired")
    slots = _slots
    if not slots.acquire(blocking=False):
        raise StageUnavailable("Investigation stage capacity is occupied")
    def run():
        token = deadline_context.set(monotonic() + seconds)
        try:
            return function(*args, **kwargs)
        finally:
            deadline_context.reset(token)
    try:
        future = _pool.submit(copy_context().run, run)
    except BaseException:
        slots.release()
        raise
    future.add_done_callback(lambda _: slots.release())
    try:
        return future.result(timeout=seconds)
    except BaseException:
        future.cancel()
        raise
