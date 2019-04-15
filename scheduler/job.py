class Job:
    def __init__(self, job_id, job_type, command, num_steps_arg, total_steps,
                 duration):
        self._job_id = job_id
        self._job_type = job_type
        self._command = command
        self._num_steps_arg = num_steps_arg
        self._total_steps = total_steps
        self._duration = duration

    @staticmethod
    def from_proto(job_proto):
        duration = None
        if job_proto.has_duration:
            duration = job_proto.duration
        return Job(job_proto.job_id, job_proto.job_type, job_proto.command,
                   job_proto.num_steps_arg, job_proto.num_steps, duration)

    @property
    def job_id(self):
        return self._job_id

    @property
    def job_type(self):
        return self._job_type

    @property
    def command(self):
        return self._command

    @property
    def num_steps_arg(self):
        return self._num_steps_arg

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def duration(self):
        return self._duration
