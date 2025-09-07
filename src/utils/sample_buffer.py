from queue import Queue, PriorityQueue
import random


class SampleBuffer:
    """
    A buffer that stores samples and allows sampling from them.
    When the buffer is full, the oldest samples are removed.
    1. Add items to the buffer with add(items: list)
    2. Sample items from the buffer with sample(k: int) -> list
    3. Check if the buffer is empty with is_empty() -> bool
    4. Check if the buffer is full with full() -> bool
    5. The buffer uses a priority queue to store items with random order.
    """
    def __init__(
        self,
        max_size: int = 1000,
    ):
        self.buffer = PriorityQueue(maxsize=max_size)

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
        priority = random.randint(0, 1000000)
        q_item = (priority, item)
        if self.buffer.full():
            self.buffer.get(block=False)
            self.buffer.put(q_item, block=False)
        else:
            self.buffer.put(q_item, block=False)

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
            return self.buffer.get(block=False)[-1] # return only the item, not the priority
        return None

    def is_empty(self):
        return self.buffer.empty()
    
    def full(self):
        return self.buffer.full()
