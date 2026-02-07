def safe_round(x, digits=4):
    if x is None:
        return None
    return round(float(x), digits)
