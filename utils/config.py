from datetime import datetime

hardware = {"nonlinear": 0.174911, "offline_linear": 3.44E-05, "online_linear": 7.01E-08}
nonlinear_ops = ['ReLU', 'PReLU', 'Hardswish', 'MaxPool']

LOG_DIR = './logs/'
# the format of the time
DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
# time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)
# the path of the log file
LOG_FILE = LOG_DIR + TIME_NOW + '.log'