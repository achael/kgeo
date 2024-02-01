# partools.py
# Wraps up some helper functions for parallel loops
# Borrowed from ehtim to avoid dependency

from builtins import str
from builtins import range
from builtins import object

from multiprocessing import Value, Lock

class Counter(object):
    """Counter object for sharing among multiprocessing jobs
    """

    def __init__(self, initval=0, maxval=0):
        self.val = Value('i', initval)
        self.maxval = maxval
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value
