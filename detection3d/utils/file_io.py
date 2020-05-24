import codecs
import importlib
import os
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