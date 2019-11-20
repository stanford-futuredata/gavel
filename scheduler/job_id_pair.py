import hashlib

class JobIdPair():

    def __init__(self, job0, job1):
        if job0 is None and job1 is None:
            raise ValueError('Cannot form JobIdPair with both ids None')
        elif job0 is None and job1 is not None:
            raise ValueError('First job id in a JobIdPair cannot be None')
        elif job0 is not None and job1 is not None:
            self._job0 = min(job0, job1)
            self._job1 = max(job0, job1)
        else:
            self._job0 = job0
            self._job1 = None

        a = self._job0
        b = self._job1
        if b is None:
            self._hash_value = a
        else:
            self._hash_value = a * a + a + b if a > b else a + b * b

        self._is_pair = self._job0 is not None and self._job1 is not None
        self._as_tuple = (self._job0, self._job1)
        if self._job1 is None:
            self._singletons = (self,)
        else:
            self._singletons = (JobIdPair(self._job0, None),
                                JobIdPair(self._job1, None))
    
        self._as_set = set([self._job0, self._job1])

        if self._job1 is None:
            self._repr = '%d' % (self._job0)
        else:
            self._repr = '(%d, %d)' % (self._job0, self._job1)

    def __getitem__(self, index):
        if index == 0:
            return self._job0
        elif index == 1:
            return self._job1
        else:
            raise ValueError('Attempting to access invalid JobIdPair '
                             'index %d' % index)

    def __lt__(self, other):
        if other._job1 is not None:
            if self._job1 is None:
                return True
            elif self._job0 == other._job0:
                return self._job1 < other._job1
        elif self._job1 is not None:
            return False
        return self._job0 < other._job0

    def __eq__(self, other):
        return self._job0 == other._job0 and self._job1 == other._job1

    def __hash__(self):
        return self._hash_value

    def __repr__(self):
        return self._repr
    
    def as_tuple(self):
        return self._as_tuple

    def as_set(self):
        return self._as_set

    def overlaps_with(self, other):
        if self._is_pair:
            raise ValueError('Can only call overlaps_with on a '
                             'single job id')
        return self._job0 in other._as_set

    def is_pair(self):
        return self._is_pair

    def singletons(self):
        return self._singletons
