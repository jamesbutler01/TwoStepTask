import time

class Timer:
    def __init__(self):
        self.start = time.time()

    def elapsedtime(self):
        elapsed_time = time.time() - self.start
        out = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        return out

