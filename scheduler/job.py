class Job:
    def __init__(self, job_id, job_type, command, working_directory,
                 num_steps_arg, total_steps, duration, scale_factor=1,
                 priority_weight=1, SLO=None, needs_data_dir=False):
        self._job_id = job_id
        self._job_type = job_type
        self._command = command
        self._working_directory = working_directory
        self._needs_data_dir = needs_data_dir
        self._num_steps_arg = num_steps_arg
        self._total_steps = total_steps
        self._duration = duration
        self._scale_factor = scale_factor
        self._priority_weight = priority_weight
        if SLO is not None and SLO < 0:
            self._SLO = None
        else:
            self._SLO = SLO

    def __str__(self):
        SLO = -1 if self._SLO is None else self._SLO
        return ('%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%f' % (
            self._job_type, self._command, self._working_directory,
            self._num_steps_arg, self._needs_data_dir, self._total_steps,
            self._scale_factor, self._priority_weight, SLO))

    @staticmethod
    def from_proto(job_proto):
        duration = None
        if job_proto.has_duration:
            duration = job_proto.duration
        return Job(job_proto.job_id, job_proto.job_type, job_proto.command,
                   job_proto.working_directory, job_proto.num_steps_arg,
                   job_proto.num_steps, duration,
                   needs_data_dir=job_proto.needs_data_dir)

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
    def working_directory(self):
        return self._working_directory

    @property
    def needs_data_dir(self):
        return self._needs_data_dir

    @property
    def num_steps_arg(self):
        return self._num_steps_arg

    @property
    def total_steps(self):
        return self._total_steps

    @total_steps.setter
    def total_steps(self, total_steps):
        self._total_steps = total_steps

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
