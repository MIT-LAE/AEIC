import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AccessRecorder:
    _paths: set[Path] = field(default_factory=set)
    enabled: bool = False

    def __post_init__(self):
        sys.addaudithook(self._hook)

    def reset(self):
        self._paths.clear()

    def start(self):
        self.enabled = True

    def stop(self):
        self.enabled = False

    @property
    def paths(self) -> list[Path]:
        return sorted(self._paths)

    def _hook(self, event, args):
        if not self.enabled:
            return

        try:
            match event:
                case 'open':
                    path, mode, *_ = args
                    skip = (
                        any(path.endswith(s) for s in ['.pyc', '.py', '.so'])
                        or any(path.startswith(p) for p in ['/etc/', '/usr/'])
                        or path.find('site-packages') >= 0
                    )
                    if 'r' in mode and not skip:
                        self._paths.add(Path(path).resolve())

                case 'sqlite3.connect':
                    self._paths.add(Path(args[0]).resolve())

        except Exception:
            pass


access_recorder = AccessRecorder()


@contextmanager
def track_file_accesses():
    access_recorder.start()
    try:
        yield
    finally:
        access_recorder.stop()
