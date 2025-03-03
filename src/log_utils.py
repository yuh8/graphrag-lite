import json
import socket
from datetime import datetime


def display_log(log_level, log_msg, detail=""):
    container_log = {}
    container_log["Level"] = str(log_level)
    container_log["TimeStamp"] = str(datetime.utcnow())
    container_log["Message"] = str(log_msg)
    container_log["Detail"] = str(detail)
    container_log["Version"] = "GRAG0.0.0"
    container_log["ContainerID"] = socket.gethostname()
    container_log["ApplicationName"] = "graphrag-lite"
    print(json.dumps(container_log), flush=True)


def display_spam(log_msg, detail=""):
    display_log("SPAM", log_msg, detail)


def display_info(log_msg, detail=""):
    display_log("INFO", log_msg, detail)


def display_warning(log_msg, detail=""):
    display_log("WARNING", log_msg, detail)


def display_error(log_msg, detail=""):
    display_log("ERROR", log_msg, detail)
