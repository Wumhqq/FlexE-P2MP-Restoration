#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @IDE     : PyCharm 
# @Project : code 
# @File    : modu_format_Al.py
# @Author  : Wumh
# @Time    : 2024/9/12 22:27
import math
def modu_format_Al(dis, band):
    TS_num = band / 5
    if dis <= 500:
        SC_cap = 25
        SC_num = math.ceil(TS_num / 5)
    else:
        SC_cap = 12.5
        SC_num = math.ceil(TS_num / 2)

    return SC_cap, SC_num

if __name__ == "__main__":
    SC_cap, SC_num = modu_format(500, 40)
    print(SC_cap, SC_num)