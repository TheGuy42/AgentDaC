import contextlib
import agentlightning as agl


class NullTracer(agl.DummyTracer):
    """
    A no-op tracer that can be used when we don't want to record any traces. 
    Inherits from `agentlightning.DummyTracer`.
    """

    def trace_context(
        self,
        name=None,
        *,
        store=None,
        rollout_id=None,
        attempt_id=None,
    ):
        return contextlib.nullcontext()

    def _trace_context_sync(
        self,
        name=None,
        *,
        rollout_id=None,
        attempt_id=None,
    ):
        return contextlib.nullcontext()

    def get_last_trace(self):
        return []
