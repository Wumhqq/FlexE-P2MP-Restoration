#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @IDE     : PyCharm 
# @Project : code 
# @File    : FS_SC.py
# @Author  : Wumh
# @Time    : 2025/6/10 23:49
def map_fs_to_sc(fs, type_):
    """
    根据type和FS值映射到对应的SC值。

    参数:
        fs (int): FS的值
        type_ (int): 类型值 (2 或 3)

    返回:
        int: 对应的SC值
    """
    if type_ == 3:
        fs_to_sc_type3 = {
            0: 0,
            1: 2,
            2: 5,
            3: 8,
            4: 12,
            5: 15
        }
        return fs_to_sc_type3.get(fs, None)
    elif type_ == 2:
        fs_to_sc_type2 = {
            0: 0,
            1: 2
        }
        return fs_to_sc_type2.get(fs, None)
    else:
        return 0  # default case when type is neither 2 nor 3



