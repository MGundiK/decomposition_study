# analysis/periods.py
def default_periods(data_name: str, freq: str):
    """
    Returns a list of seasonal periods (in samples) for the dataset.
    """
    data = data_name.lower()
    if data in {"etth1", "etth2", "electricity", "traffic", "weather"} and freq in {"h", "hour", "hourly"}:
        return [24, 168]  # day, week
    if data in {"ettm1", "ettm2"}:            # 15-min
        return [96, 96*7]  # day, week
    if data in {"exchange"} or ("exchange" in data):
        return [5, 7]      # weak weekly; business-week signal shows at ~5
    if data in {"illness", "ili"}:
        return [52]        # weekly seasonality in yearly units
    if data in {"solar"}:
        return [24]        # daily dominant
    # Fallback
    return [24]
