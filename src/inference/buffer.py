import threading

import numpy as np


class CircularBuffer:
    def __init__(self, size: int, dtype=float):
        self.size = size
        self.buffer = np.zeros(size, dtype=dtype)
        self.head = 0
        self.count = 0
        self.lock = threading.Lock()

    def put(self, item):
        with self.lock:
            if self.count < self.size:
                self.buffer[self.head] = item
                self.count += 1
            else:
                self.buffer[self.head] = item

            self.head = (self.head + 1) % self.size

    def put_many(self, items):
        with self.lock:
            n = len(items)
            first_chunk = min(n, self.size - self.head)
            self.buffer[self.head : self.head + first_chunk] = items[:first_chunk]
            if first_chunk < n:
                self.buffer[: n - first_chunk] = items[first_chunk:]
            self.head = (self.head + n) % self.size
            self.count = min(self.count + n, self.size)

    def get(self):
        with self.lock:
            if self.count == 0:
                return np.array([])

            if self.count < self.size:
                return self.buffer[: self.count]

            return np.concatenate((self.buffer[self.head :], self.buffer[: self.head]))
