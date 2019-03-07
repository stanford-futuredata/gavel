import threading

"""
Implementation of a thread-safe queue.
"""
class Queue:
    def __init__(self):
        self.queue = []
        self.cv = threading.Condition()

    def add(self, key, item):
        self.cv.acquire()
        self.queue.append((key, item))
        self.queue.sort(key=lambda x: x[0])
        self.cv.notify()
        self.cv.release()

    def remove(self):
        self.cv.acquire()
        while len(self.queue) == 0:
            self.cv.wait()
        key, item = self.queue.pop(0)
        self.cv.release()
        return key, item
