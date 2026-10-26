"""Save a run's console output to a file without changing what the console shows.

``start_run_log()`` mirrors everything the process writes to stdout and stderr
into ``out/logs/run.log``. The terminal still shows it all live; the file is a
copy, so a crash can be read back afterwards instead of being scrolled for.

The redirection happens at the OS file-descriptor level rather than by replacing
``sys.stdout``, because the two most useful sources bypass ``sys.stdout``
entirely: ``_evaluate_epoch()`` runs NNEval through
``subprocess.run(cmd, check=True, env=env)``, which hands the child fd 1 and 2
directly, and torch writes its warnings (``[W1018 ...] AllocatorConfig``) from
C++. A ``sys.stdout`` wrapper would capture neither.

Previous logs are deleted on every start, so the directory always holds exactly
one run -- the current one. The log lives OUTSIDE ``out/nngpt``, which
``reset_run_artifacts`` / ``reset_run_dir`` clears at the start of a run, so the
reset cannot delete the log that is recording it.
"""

import atexit
import datetime
import os
import sys
import threading
from pathlib import Path

from ab.nn.util.Const import out_dir

LOG_NAME = 'run.log'

_started = None


def log_dir_path(log_dir=None):
    """Where the run log lives. Explicit argument wins, then NNGPT_LOG_DIR."""
    if log_dir:
        return Path(log_dir)
    env = os.environ.get('NNGPT_LOG_DIR', '').strip()
    return Path(env) if env else Path(out_dir) / 'logs'


def clear_logs(log_dir=None):
    """Delete the previous run's logs. Returns how many were removed."""
    directory = log_dir_path(log_dir)
    if not directory.is_dir():
        return 0
    removed = 0
    for stale in directory.glob('*.log'):
        try:
            stale.unlink()
            removed += 1
        except Exception as exc:
            print(f'[RUN LOG] Could not remove {stale}: {exc}')
    return removed


def start_run_log(log_dir=None, clear=True):
    """Begin mirroring stdout and stderr to the run log. Returns its path.

    Safe to call more than once: the second call is a no-op and returns the
    path already in use. Never raises -- a logging failure must not stop a run.
    """
    global _started
    if _started is not None:
        return _started

    try:
        directory = log_dir_path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        removed = clear_logs(directory) if clear else 0
        path = directory / LOG_NAME
        handle = open(path, 'wb', buffering=0)
    except Exception as exc:
        print(f'[RUN LOG] Disabled, could not open the log file: {exc}')
        return None

    saved_out, saved_err = os.dup(1), os.dup(2)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def _pump():
        while True:
            try:
                chunk = os.read(read_fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            try:
                os.write(saved_out, chunk)
                handle.write(chunk)
            except Exception:
                break

    thread = threading.Thread(target=_pump, name='run-log', daemon=True)
    thread.start()

    # fd 1 is a pipe now, so Python switches stdout from line buffering to block
    # buffering and the console goes silent for minutes at a time. Put it back.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass

    def _finish():
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        try:
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(read_fd)
        except OSError:
            pass
        thread.join(timeout=5)
        try:
            handle.close()
        except Exception:
            pass

    atexit.register(_finish)
    _started = path

    print(f'[RUN LOG] {path}'
          + (f' (cleared {removed} log file(s) from the previous run)' if removed else ''))
    print(f'[RUN LOG] started {datetime.datetime.now():%Y-%m-%d %H:%M:%S}')
    return path
