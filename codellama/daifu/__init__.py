import sys

python_version = sys.version_info
if python_version.major == 3 and python_version.minor >= 11:
    from .goto311 import with_goto
elif python_version.major == 3 and python_version.minor >= 8:
    from .goto38 import with_goto
else:
    from .goto import with_goto

from .context import CT_MANAGER
from .repair import RP_MANAGER
from .transform import transform, TRANSFORM_REGISTRY

#from .liveupdate import Update, UpdateManager, Redefine, Instrument

from loguru import logger
from pathlib import Path
import time
import shutil

import multiprocessing

import signal

logger.add("experiment_record.log")

if multiprocessing.current_process().name in ["MainProcess", "MainProcess-2"]:
    log_file = Path.cwd()/"experiment_record.log"

    log_lock_file = Path.cwd()/"experiment_record.log.lock"
    if not log_lock_file.exists():
        with log_file.open('w') as f:
            f.write('')

        logger.info("Program Begin")


def init_preemptation_handler(preemptation_handler):
    signal.signal(signal.SIGTERM, preemptation_handler)
    import os
    print(os.getpid())


def log_program_end(value='', folder=None, milestone=False):
    if multiprocessing.current_process().name in ["MainProcess", "SpawnProcess-2", "MainProcess-2"]:
        if value != '':
            value = ' with Quality Measure ' + str(value)
        logger.info("Program End"+value)
        if 'REPEAT_ACTIONS' in TRANSFORM_REGISTRY or milestone:
            log_file = Path.cwd()/"experiment_record.log"
            if folder is not None:
                (Path.cwd()/folder).mkdir(parents=True, exist_ok=True)
                repeat_file = Path.cwd()/folder/("experiment_record_"+str(time.time())+".log")
            else:
                repeat_file = Path.cwd()/("experiment_record_"+str(time.time())+".log")
            shutil.copyfile(log_file, repeat_file)

        try:
            log_lock_file = Path.cwd()/"experiment_record.log.lock"
            log_lock_file.unlink(missing_ok = True)
        except:
            log_lock_file = Path.cwd()/"experiment_record.log.lock"
            if log_lock_file.exists():
                log_lock_file.unlink()
        