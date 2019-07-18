import heapq

class JobQueue:
    
    class JobQueueEntry(object):

        def __init__(self, priority, allocation, steps_run, job_id):
            self._priority = priority
            self._allocation = allocation
            self._steps_run = steps_run
            self._job_id = job_id

        def __repr__(self):
            return "Job ID: %s, Priority: %f" % (self._job_id, self._priority)

        @property
        def priority(self):
            return self._priority

        @priority.setter
        def priority(self, priority):
            self._priority = priority

        @property
        def allocation(self):
            return self._allocation
        
        @allocation.setter
        def allocation(self, allocation):
            self._allocation = allocation

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
            elif self._allocation != other._allocation:
                return self._allocation < other._allocation
            elif self._steps_run != other._steps_run:
                return self._steps_run < other._steps_run
            else:
                return self._job_id < other._job_id

        def __eq__(self, other):
            return (self._priority == other_.priority
                    and self._allocation == other._allocation
                    and self._steps_run == other._steps_run
                    and self._job_id == other._job_id)

    def __init__(self):
        self._queue = []

    def __getitem__(self, index):
        return self._queue[index]

    def add_job(self, priority, allocation, steps_run, job_id, heappush=False):
        entry = self.JobQueueEntry(priority, allocation, steps_run, job_id)
        if heappush:
            heapq.heappush(self._queue, entry)
        else:
            self._queue.append(entry)

    def pop(self, i):
        self._queue.pop(i)

    def heapify(self):
        heapq.heapify(self._queue)

    def update_entry(self, i, priority=None, allocation=None,
                     steps_run=None):
        if priority is not None:
            self._queue[i].priority = priority

        if allocation is not None:
            self._queue[i].allocation = allocation

        if steps_run is not None:
            self._queue[i].steps_run = steps_run

    def size(self):
        return len(self._queue)

    def get_sorted_queue(self):
        return sorted(self._queue)
