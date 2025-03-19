import datetime

FMT = "%(asctime)s:ha-hps: %(levelname)s - %(message)s"
TIMEFMT = "%Y-%m-%d %H:%M:%S"

# Record the date in the format YYYY-MM-DD for use in filenames
DATESTR = datetime.datetime.now().strftime("%Y-%m-%d")
