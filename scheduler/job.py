class Job:
    def __init__(self, job_id, command, num_steps, duration):
        self._job_id = job_id
        self._command = command
        self._num_steps = num_steps
        self._duration = duration

    @staticmethod
    def from_proto(job_proto):
        duration = None
        if job_proto.has_duration:
            duration = job_proto.duration
        return Job(job_proto.job_id, job_proto.command,
                   job_proto.num_steps, duration) 

    def job_id(self):
        return self._job_id

    def command(self):
        return self._command

    def num_steps(self):
        return self._num_steps

    def duration(self):
        return self._duration
