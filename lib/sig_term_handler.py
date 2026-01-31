import signal

class SigTermHandler:
  raised = False
  callback = None

  def __init__(self, callback=None):
    self.callback = callback
    signal.signal(signal.SIGTERM, self.handler)
    signal.signal(signal.SIGINT, self.handler)

  def handler(self, signal, frame):
    self.raised = True
    if self.callback:
      self.callback(signal)
