from datetime import datetime, timedelta

def get_fridays(start_date: str, end_date: str) -> list[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    #判断start是否是周五，如不是，则往后推，直到找到第一个周五
    while start.weekday() != 4 and start <= end:
        start += timedelta(days=1)
    
    fridays = []
    while start <= end:
        fridays.append(start.strftime("%Y-%m-%d"))
        start += timedelta(days=7)

    return fridays


if __name__ == "__main__":
    start_date = "2024-01-01"
    end_date = "2026-03-31"
    fridays = get_fridays(start_date, end_date)
    print(fridays)
