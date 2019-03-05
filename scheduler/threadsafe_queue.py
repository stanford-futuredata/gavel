import threading

"""
Implementation of a thread-safe queue.
"""
class Queue:
    def __init__(self):
        self.queue = []
        self.cv = threading.Condition()

    def add(self, item):
        self.cv.acquire()
        self.queue.append(item)
        self.cv.notify()
        self.cv.release()

    def remove(self):
        self.cv.acquire()
        while len(self.queue) == 0:
            self.cv.wait()
        item = self.queue.pop(0)
        self.cv.release()
        return item
