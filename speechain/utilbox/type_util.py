"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.08
"""
def str2bool(str: str) -> bool:
    if str.lower() == 'true':
        return True
    elif str.lower() == 'false':
        return False
    else:
        raise ValueError