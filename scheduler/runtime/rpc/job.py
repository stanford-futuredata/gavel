class Job:
  def __init__(self, job_proto):
    self._job_id = job_proto.job_id
    self._command = job_proto.command
    self._num_epochs = job_proto.num_epochs

  def job_id(self):
    return self._job_id

  def command(self):
    return self._command

  def num_epochs(self):
    return self._num_epochs
