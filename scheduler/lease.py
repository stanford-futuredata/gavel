class Lease:
    def __init__(self, max_steps, max_duration):
        self._max_steps = max_steps
        self._max_duration = max_duration

    def __str__(self):
        return 'max_steps: %d, max_duration: %f' % (self._max_steps,
                                                    self._max_duration)

    @property
    def max_steps(self):
        return self._max_steps

    @max_steps.setter
    def max_steps(self, steps):
        self._max_steps = steps

    @property
    def max_duration(self):
        return self._max_duration

    @max_duration.setter
    def max_duration(self, duration):
        self._max_duration = duration
