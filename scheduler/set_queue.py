import queue

class SetQueue(queue.Queue):
    """Create a queue object with a set as the underlying data structure."""
    def _init(self, maxsize):
        self.queue = set()

    def _put(self, item):
        self.queue.add(item)

    def _get(self):
        return self.queue.pop()

    def __contains__(self, item):
        with self.mutex:
            return item in self.queue
