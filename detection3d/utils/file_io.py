import codecs
import importlib
import os
import re
import sys
import logging


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


def get_next_run_dir(base_dir: str) -> str:
    """
    Find the next run directory name in base_dir.
    E.g. if run_1..run_7 exist, returns run_8.
    """
    os.makedirs(base_dir, exist_ok=True)  # Ensure base dir exists

    # List all dirs starting with 'run_'
    run_dirs = [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run")]

    # Extract run numbers using regex
    run_nums = []
    for d in run_dirs:
        match = re.match(r"run[_-]?(\d+)", d)
        if match:
            run_nums.append(int(match.group(1)))

    next_num = max(run_nums) + 1 if run_nums else 1
    next_run_name = f"run_{next_num}"
    next_run_path = os.path.join(base_dir, next_run_name)

    os.makedirs(next_run_path, exist_ok=True)  # Create the new folder
    return next_run_path
