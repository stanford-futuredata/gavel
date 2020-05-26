
class JobTypeSet():

    def __init__(self, job_types):
        self._job_types = tuple(sorted(job_types))

    def __eq__(self, other):
        return self._job_types == other._job_types

    def __hash__(self):
        return hash(self._job_types)

    def __str__(self):
        return str(self._job_types)

    def __getitem__(self, item):
        return self._job_types[item]

    def __len__(self):
        return len(self._job_types)
