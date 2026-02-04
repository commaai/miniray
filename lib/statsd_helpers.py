import statsd
import atexit

global_tags: dict[str, str] = {}

def with_tags(stat, tags={}):
  return stat + ''.join(f',{k}={v}' for k, v in (global_tags | tags).items())


class StatsClient(statsd.StatsClient):
  def bind_global(self, **kwargs):
    global_tags.update(kwargs)

  def event(self, stat, tags={}, **kwargs):
    self.incr(stat, tags=tags)
    for key, value in kwargs.items():
      self.hist(f'{stat}.{key}', value=value, tags=tags)

  def hist(self, stat, value, rate=1, tags={}):
    self._send_stat(with_tags(stat, tags), '%0.6f|h' % value, rate)

  # overrides
  def timing(self, stat, delta, rate=1, tags={}):
    super().timing(with_tags(stat, tags), delta, rate=rate)

  def incr(self, stat, count=1, rate=1, tags={}):
    super().incr(with_tags(stat, tags), count=count, rate=rate)

  def decr(self, stat, count=1, rate=1, tags={}):
    super().decr(with_tags(stat, tags), count=count, rate=rate)

  def gauge(self, stat, values, rate=1, delta=False, tags={}):
    super().gauge(with_tags(stat, tags), values, rate=rate, delta=delta)

  def set(self, stat, value, rate=1, tags={}):
    super().set(with_tags(stat, tags), value, rate=rate)


statsd = StatsClient('localhost', 8125)
atexit.register(lambda: statsd.close())
