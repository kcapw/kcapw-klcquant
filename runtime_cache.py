from __future__ import annotations

from collections import OrderedDict

import torch

from .streamed_loader import LoadedTensorGroup


class RuntimeTensorCache:
    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.groups: OrderedDict[str, LoadedTensorGroup] = OrderedDict()

    def get(self, group: str) -> LoadedTensorGroup | None:
        item = self.groups.get(group)
        if item is not None:
            self.groups.move_to_end(group)
        return item

    def put(self, item: LoadedTensorGroup) -> None:
        if item.group in self.groups:
            self.current_bytes -= self.groups[item.group].nbytes
            self.groups[item.group].unload()
        self.groups[item.group] = item
        self.current_bytes += item.nbytes
        self.groups.move_to_end(item.group)
        self.evict()

    def evict(self) -> None:
        while self.current_bytes > self.max_bytes and self.groups:
            _, item = self.groups.popitem(last=False)
            self.current_bytes -= item.nbytes
            item.unload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear(self) -> None:
        for item in self.groups.values():
            item.unload()
        self.groups.clear()
        self.current_bytes = 0
