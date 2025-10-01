import time, datetime as dt

def sleep_to_next_minute(offset_sec=2):
    now = dt.datetime.now(dt.timezone.utc)
    next_min = (now.replace(second=0, microsecond=0) + dt.timedelta(minutes=1))
    target = next_min + dt.timedelta(seconds=offset_sec)
    delta = (target - now).total_seconds()
    if delta>0: time.sleep(delta)
