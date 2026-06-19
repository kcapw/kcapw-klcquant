from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

from .quantized_runtime_loader import LoadedRuntimeGroup, QuantizedRuntimeLoader


class RuntimePrefetcher:
    def __init__(self, loader: QuantizedRuntimeLoader, max_workers: int = 1) -> None:
        self.loader = loader
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._future: Future[LoadedRuntimeGroup] | None = None
        self._group: str | None = None

    def prefetch(self, group: str) -> None:
        self._group = group
        self._future = self.executor.submit(self.loader.load_group, group)

    def consume(self) -> LoadedRuntimeGroup | None:
        if self._future is None:
            return None
        result = self._future.result()
        self._future = None
        self._group = None
        return result

    @property
    def pending_group(self) -> str | None:
        return self._group
