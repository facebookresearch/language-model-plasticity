# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

""" Some helpful function on visualisation"""
def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

def percentage(f):
    p = f * 100
    return f'{int(p)}%' if p == int(p) else f'{p}%'


def invert_human_format(s):
    """
    Inverts the human_format function by converting a string like '1.23M' to the corresponding number (1230000).
    """
    s = s.strip()
    if s[-1] == 'K':
        return float(s[:-1]) * 1000
    elif s[-1] == 'M':
        return float(s[:-1]) * 1000000
    elif s[-1] == 'B':
        return float(s[:-1]) * 1000000000
    elif s[-1] == 'T':
        return float(s[:-1]) * 1000000000000
    else:
        return float(s)
