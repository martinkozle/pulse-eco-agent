import datetime

from langchain.tools import tool


@tool("now_plus_hours")
def now_plus_hours(hours: int) -> str:
    """Add a number of hours to now, supports negative hours."""
    hours = int(hours)
    return (
        datetime.datetime.now(tz=datetime.UTC) + datetime.timedelta(hours=hours)
    ).isoformat()


@tool("datetime_plus_days")
def datetime_plus_days(days: int) -> str:
    """Add a number of days to now, supports negative days."""
    days = int(days)
    return (
        datetime.datetime.now(tz=datetime.UTC) + datetime.timedelta(days=days)
    ).isoformat()
