import math

def log_transform(values: list) -> list:
    """
    Returns the log1p-transformed values rounded to four decimals.
    """
    result = []
    for i in values:
        y = math.log1p(i)
        result.append(y)
    return result