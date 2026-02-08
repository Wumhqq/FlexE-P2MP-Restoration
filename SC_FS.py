#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @IDE     : PyCharm 
# @Project : code 
# @File    : SC_FS.py
# @Author  : Wumh
# @Time    : 2025/6/9 17:44
def sc_fs(hub_type, sc_index, start_or_end):
    """
    计算某个子载波 (SC) 所属的相对 FS 序号。

    Parameters
    ----------
    hub_type : int
        Hub 的类别 (1、2 或 ≥3)。
    sc_index : int
        子载波序号，范围 1–16。
    start_or_end : int
        1 表示 sc_index 是区段的“起始” SC，
        0（或其它值）表示 sc_index 是“终止” SC。

    Returns
    -------
    int
        计算得到的相对 FS 序号（仅用于统计占用 FS 的个数，
        并非真实网络中的绝对 FS 编号）。
    """
    # --- Hub type 1 ------------------------------------------------------
    if hub_type == 1:
        return 0

    # --- Hub type 2 ------------------------------------------------------
    if hub_type == 2:
        return 0 if sc_index <= 1 else 1

    # --- Hub type ≥3 -----------------------------------------------------
    # 基本区间映射
    if sc_index <= 1:
        fs_idx = 0
    elif sc_index <= 4:
        fs_idx = 1
    elif sc_index <= 7:
        fs_idx = 2
    elif sc_index <= 11:
        fs_idx = 3
    elif sc_index <= 14:
        fs_idx = 4
    else:                               # sc_index == 16
        fs_idx = 5

    # 起始 / 终止边界微调
    if start_or_end == 1:               # 起始 SC
        if sc_index == 1:
            fs_idx = 0
        elif sc_index == 4:
            fs_idx = 1
        elif sc_index == 11:
            fs_idx = 3
        elif sc_index == 14:
            fs_idx = 4
    else:                               # 终止 SC
        if sc_index == 1:
            fs_idx = 1
        elif sc_index == 4:
            fs_idx = 2
        elif sc_index == 11:
            fs_idx = 4
        elif sc_index == 14:
            fs_idx = 5

    return fs_idx
