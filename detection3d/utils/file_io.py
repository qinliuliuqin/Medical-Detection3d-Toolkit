import codecs
import importlib
import os
import re
import sys
import logging
from typing import Optional

def load_config(pyfile):
    """
    load configuration file as a python module
    :param pyfile     configuration python file
    :return a loaded network module
    """
    assert os.path.isfile(pyfile), 'The file {} does not exits!'.format(pyfile)

    dirname = os.path.dirname(pyfile)
    basename = os.path.basename(pyfile)
    modulename, _ = os.path.splitext(basename)

    need_reload = modulename in sys.modules

    os.sys.path.append(dirname)
    config = importlib.import_module(modulename)
    if need_reload:
        importlib.reload(config)
    del os.sys.path[-1]

    return config.cfg


def setup_logger(log_file, name):
    """
    setup logger for logging training messages
    :param log_file: the location of log file
    :param name: the name of logger
    :return: a logger object
    """
    dirname = os.path.dirname(log_file)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # stream handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def readlines(file):
    """
    read lines by removing '\n' in the end of line
    :param file: a text file
    :return: a list of line strings
    """
    fp = codecs.open(file, 'r', encoding='utf-8')
    linelist = fp.readlines()
    fp.close()
    for i in range(len(linelist)):
        linelist[i] = linelist[i].rstrip('\n')
    return linelist


def get_run_dir(base_dir: str, mode: str = "next") -> Optional[str]:
    """
    Get either the current/latest run directory or the next run directory.

    Args:
        base_dir (str): Path to the base directory where run_* folders are stored.
        mode (str): "current" for the latest run, "next" for the next run (default: "next").

    Returns:
        str | None: Path to the run directory.
                    Returns None if mode="current" and no run exists.
    """
    # Collect existing run directories (if base_dir exists)
    run_nums = []
    if os.path.exists(base_dir):
        run_dirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run")
        ]

        for d in run_dirs:
            match = re.match(r"run[_-]?(\d+)", d)
            if match:
                run_nums.append((int(match.group(1)), d))

    if mode == "current":
        if not run_nums:
            return None
        latest_num, latest_dir = max(run_nums, key=lambda x: x[0])
        return os.path.join(base_dir, latest_dir)

    elif mode == "next":
        os.makedirs(base_dir, exist_ok=True)
        next_num = max([n for n, _ in run_nums], default=0) + 1
        next_run_name = f"run_{next_num}"
        next_run_path = os.path.join(base_dir, next_run_name)
        os.makedirs(next_run_path, exist_ok=True)
        return next_run_path

    else:
        raise ValueError("mode must be 'current' or 'next'")


def get_resolved_run_dir(path: str) -> str:
    """
    Return the path to a valid run directory.
    
    - If `path` already ends with run_{number}, return it.
    - Otherwise, return the latest run_* inside `path`.
    - Raise FileNotFoundError if no run directories are found.
    """
    norm = os.path.normpath(path)

    # If already a run_{n} folder, return as-is
    if re.search(r"(?:^|[/\\])run[_-]\d+$", norm):
        return norm

    # Otherwise, resolve to latest run
    run_dir: Optional[str] = get_run_dir(path, mode="current")
    if run_dir is None:
        raise FileNotFoundError(f"No run_* directory found in: {path}")

    return run_dir
