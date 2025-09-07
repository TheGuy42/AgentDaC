from queue import Queue


class SampleBuffer:
    def __init__(
        self,
        max_size: int = 1000,
    ):
        self.buffer = Queue(maxsize=max_size)

    def add(self, items: list) -> None:
        """
        Add multiple items to the buffer.
        If the buffer is full, the oldest items will be removed.
        """
        for item in items:
            self._add(item)
    
    def _add(self, item):
        """
        Add a single item to the buffer.
        If the buffer is full, the oldest item will be removed.
        """
        if self.buffer.full():
            self.buffer.get(block=False)
            self.buffer.put(item, block=False)
        else:
            self.buffer.put(item, block=False)

    def sample(self, k: int=1) -> list:
        """
        Sample k items from the buffer.
        If the buffer has less than k items, return all items.
        """
        samples = []
        for _ in range(k):
            sample = self._sample()
            if sample is not None:
                samples.append(sample)
        return samples

    def _sample(self):
        if not self.buffer.empty():
            return self.buffer.get(block=False)
        return None

    def is_empty(self):
        return self.buffer.empty()
    
    def full(self):
        return self.buffer.full()
