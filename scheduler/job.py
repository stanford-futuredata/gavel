class Job:
    def __init__(self, job_id, job_type, command, num_steps_arg, total_steps,
                 duration, scale_factor=1, priority_weight=1, SLO=None):
        self._job_id = job_id
        self._job_type = job_type
        self._command = command
        self._num_steps_arg = num_steps_arg
        self._total_steps = total_steps
        self._duration = duration
        self._scale_factor = scale_factor
        self._priority_weight = priority_weight
        if SLO is not None and SLO < 0:
            self._SLO = None
        else:
            self._SLO = SLO

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

    @property
    def scale_factor(self):
        return self._scale_factor

    @property
    def priority_weight(self):
        return self._priority_weight

    @property
    def SLO(self):
        return self._SLO
