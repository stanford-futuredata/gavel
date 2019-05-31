import hashlib

class JobIdPair():

    def __init__(self, job0, job1):
        if job0 is None and job1 is None:
            raise ValueError('Cannot form JobIdPair with both ids None')
        elif job0 is None and job1 is not None:
            raise ValueError('First job id in a JobIdPair cannot be None')
        self._job0 = job0
        self._job1 = job1

        a = self._job0
        b = self._job1
        if b is None:
            self._hash_value = a
        else:
            self._hash_value = a * a + a + b if a > b else a + b * b

    def __getitem__(self, index):
        if index == 0:
            return self._job0
        elif index == 1:
            return self._job1
        else:
            raise ValueError('Attempting to access invalid JobIdPair '
                             'index %d' % index)

    def __lt__(self, other):
        if self._job0 != other._job0:
            return self._job0 < other._job0
        elif self._job1 is None and self._job0 is None:
            return False
        elif self._job1 is not None and other._job1 is not None:
            return self._job0 < other._job1
        else:
            return self._job1 is None

    def __eq__(self, other):
        return self._job0 == other._job0 and self._job1 == other._job1

    def __hash__(self):
        return self._hash_value

    def __repr__(self):
        if self[1] is None:
            return '%d' % (self[0])
        else:
            return ('(%d, %d)' % (self[0], self[1]))

    def as_tuple(self):
        return (self._job0, self._job1)

    def overlaps_with(self, other):
        if self.is_pair():
            raise ValueError('Can only call overlaps_with on a '
                             'single job id')
        return ((other[0] is not None and self[0] == other[0]) or
                (other[1] is not None and self[0] == other[1]))

    def is_pair(self):
        return self._job0 is not None and self._job1 is not None

    def singletons(self):
        if self[1] is None:
            return (self,)
        else:
            return (JobIdPair(self[0], None),
                    JobIdPair(self[1], None))
