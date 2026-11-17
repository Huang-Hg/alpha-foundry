"""Small owned feedback leases, independent of mutable portfolio state."""
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType

from foundry.types import ComputeDevice, FloatArray, HostArray


class FeedbackVector:
    """Take ownership of a finite FP32 vector; never retain a market panel.

    One consumer may borrow at a time. A CUDA event captures the producing
    stream even when consumption happens in another Python thread. Consumers
    must not mutate the view or keep aliases beyond the borrow scope.
    """

    def __init__(self, values: FloatArray, xp: ModuleType, device: ComputeDevice) -> None:
        if values.ndim != 1 or values.dtype != xp.float32:
            raise ValueError('feedback requires a contiguous FP32 vector')
        if not values.flags.c_contiguous:
            raise ValueError('feedback requires a contiguous FP32 vector')
        self.device, self.shape, self.xp = device, values.shape, xp
        self._values: FloatArray | None = values
        self._borrowing = False
        self._closed = False
        self._ready = None
        self._consumed = None
        if device.kind == 'cuda':
            self._ready = xp.cuda.Event()
            self._ready.record()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError('feedback lease is closed')

    @contextmanager
    def borrow(self, stream: int | None = None) -> Iterator[FloatArray]:
        self._require_open()
        if self._borrowing:
            raise RuntimeError('feedback already has a consumer')
        if (stream is None) != (self.device.kind == 'cpu'):
            raise ValueError('CUDA feedback requires an explicit consumer stream')
        self._borrowing = True
        try:
            if self.device.kind == 'cuda':
                with self.xp.cuda.Device(self.device.index), self.xp.cuda.ExternalStream(stream):
                    self.xp.cuda.get_current_stream().wait_event(self._ready)
                    try:
                        yield self._values
                    finally:
                        self._consumed = self.xp.cuda.Event()
                        self._consumed.record()
            else:
                view = self._values.view()
                view.flags.writeable = False
                yield view
        finally:
            self._borrowing = False

    def to_host(self) -> HostArray:
        """Explicit report/callback materialization; never used by GPU learning."""
        self._require_open()
        if self.device.kind == 'cuda':
            self._ready.synchronize()
            with self.xp.cuda.Device(self.device.index):
                return self.xp.asnumpy(self._values)
        return self._values.copy()

    def close(self) -> None:
        if self._borrowing:
            raise RuntimeError('cannot close feedback while its consumer is borrowing')
        if self._ready is not None:
            self._ready.synchronize()
        if self._consumed is not None:
            self._consumed.synchronize()
        self._values = None
        self._closed = True
