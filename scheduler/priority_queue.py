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

    def remove_item(self, item_to_remove):
        self.cv.acquire()
        while len(self.queue) == 0:
            self.cv.wait()
        for i in range(len(self.queue)):
            if self.queue[i][1] == item_to_remove:
                key, item = self.queue.pop(i)
                self.cv.release()
                return key, item
        self.cv.release()
        return None, None
