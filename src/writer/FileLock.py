import time
import os


class FileLock:

    def __init__(self, path, backoff_delay=.5):
        self._path = path
        self._backoff_delay = backoff_delay
        self._has_lock = False

    def acquire(self, timeout=0):
        began_at = time.time()
        elapsed = time.time() - began_at
        while not self._has_lock:
            try:
                with open(self._path, "x"):
                    self._has_lock = True
                    break
            except FileExistsError:
                elapsed = time.time() - began_at
                if timeout > 0 and elapsed > timeout:
                    raise TimeoutError(f"Couldn't acquire lock on {self._path}.")
                time.sleep(self._backoff_delay)

    def release(self):
        if self._has_lock:
            os.remove(self._path)
            self._has_lock = False
        else:
            raise RuntimeError(f"Can't release lock on {self._path}, "
                               "it's not mine.")

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, type, value, traceback):
        self.release()



