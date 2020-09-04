import datetime
import logging

class SchedulerAdapter(logging.LoggerAdapter):

    def process(self, msg, kwargs):
        scheduler = self.extra['scheduler']
        timestamp = scheduler.get_current_timestamp(in_seconds=True)
        timestamp = (self.extra['start_timestamp'] +
                     datetime.timedelta(0, timestamp))
        return '[%s] %s' % (timestamp, msg), kwargs
