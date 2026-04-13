import sys
import time

from tqdm import tqdm


class Progress:
    def __init__(self, total, desc=None, log_interval=10):
        self.total = total
        self.desc = desc or ''

        if total == 0:
            self.noop = True
            return
        self.noop = False

        self.is_tty = sys.stderr.isatty()

        if self.is_tty:
            self.pbar = tqdm(total=total, desc=desc)
        else:
            self.count = 0
            self.last_log = time.time()
            self.log_interval = log_interval

    def update(self, n=1):
        if self.noop:
            return
        if self.is_tty:
            self.pbar.update(n)
        else:
            self.count += n
            now = time.time()
            if now - self.last_log > self.log_interval:
                pct = 100.0 * self.count / self.total
                print(f'{self.desc}: {pct:6.2f}% ({self.count}/{self.total})')
                self.last_log = now

    def close(self):
        if self.noop:
            return
        if self.is_tty:
            self.pbar.close()
        else:
            # final log
            pct = 100.0 * self.count / self.total
            print(f'{self.desc}: {pct:6.2f}% ({self.count}/{self.total}) DONE')
