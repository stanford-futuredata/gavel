class JobQueue:
    
    class JobQueueEntry(object):

        def __init__(self, priority, steps_run, job_id):
            self._priority = priority
            self._steps_run = steps_run
            self._job_id = job_id

        @property
        def priority(self):
            return self._priority

        @priority.setter
        def priority(self, priority):
            self._priority = priority

        @property
        def steps_run(self):
            return self._steps_run

        @steps_run.setter
        def steps_run(self, steps_run):
            self._steps_run = steps_run

        @property
        def job_id(self):
            return self._job_id

        def __lt__(self, other):
            if self._priority != other._priority:
                return self._priority < other._priority
            elif self._steps_run != other._steps_run:
                return self._steps_run < other.steps_run
            else:
                return self._job_id < other.job_id

        def __eq__(self, other):
            return (self._priority == other_.priority
                    and self._steps_run == other._steps_run
                    and self._job_id == other._job_id)

    def __init__(self):
        self._queue = []

    def __getitem__(self, index):
        return self._queue[index]

    def add_job(self, priority, steps_run, job_id, heappush=False):
        entry = self.JobQueueEntry(priority, steps_run, job_id)
