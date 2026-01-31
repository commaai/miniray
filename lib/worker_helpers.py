import time


class ExponentialBackoff:
  def __init__(self, max_st, debug=False):
    self.max_st = max_st
    self.debug = debug
    self.reset()

  def reset(self):
    self.no_task_cnt = 0

  def sleep(self):
    if self.no_task_cnt > 0:
      st = min(2 ** self.no_task_cnt / 10, self.max_st) # exponential backoff
      if self.debug: print("[worker] no task sleep:", st)
      time.sleep(st)
    self.no_task_cnt = min(self.no_task_cnt+1, 10) # limit value because it is used as an exponent
