#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @IDE     : PyCharm 
# @Project : code 
# @File    : test_right.py
# @Author  : Wumh
# @Time    : 2024/9/1 21:43
import numpy as np
import math


def test_right(node_P2MP, node_flow, flow_acc, flows_num, topo_num, Tbox_num, Tbox_P2MP, type):
    # 1.检测T-box容量限制，P2MP的SC限制（收发）
    # 2.检测是否每条流都有scr和des的收发机安排
    flow_count = np.zeros((flows_num, 3), dtype=int)
    type_P2MP = [[1, 1], [2, 4], [3, 16]]

    for i in range(flows_num):
        scr = flow_acc[i][1]
        des = flow_acc[i][2]
        hub = flow_acc[i][7]
        leaf = flow_acc[i][8]
        flow_count[i][0] = i
        flag = 0
        for f in node_flow[scr][hub][0]:
            if f[0] == flow_acc[i][0]:
                flag = 1
        if flag == 0:
            print("flow_acc: flow %d 所在的 node %d hub %d 并没有该条流，可能是P2MP不充足的原因，建议检查一下" %(i, scr, hub))

        flag = 0
        for f in node_flow[des][leaf][0]:
            if f[0] == flow_acc[i][0]:
                flag = 1
        if flag == 0:
            print("flow_acc: flow %d 所在的 node %d leaf %d 并没有该条流，可能是P2MP不充足的原因，建议检查一下" %(i, des, leaf))

    for n in range(topo_num):
        for p in range(Tbox_num * Tbox_P2MP):
            if node_P2MP[n][p][2] != 0:
                if type == 1 and len(node_flow[n][p][0]) == 0:
                    print("node_P2MP: node %d P2MP %d 被标记了使用但是该位置并没有流量" %(n, p))
                elif len(node_flow[n][p][0]) != 0:
                    for f in node_flow[n][p][0]:
                        if node_P2MP[n][p][2] == 1 or node_P2MP[n][p][2] == 3:
                            flow_count[f[0]][1] = flow_count[f[0]][1] + 1
                        elif node_P2MP[n][p][2] == 2 or node_P2MP[n][p][2] == 4:
                            flow_count[f[0]][2] = flow_count[f[0]][2] + 1
            else:
                if len(node_flow[n][p][0]) != 0:
                    print("node_P2MP: node %d P2MP %d 未被标记了使用但是该位置有流量" %(n, p))

    for i in range(flows_num):
        if flow_count[i][1] == 0 or flow_count[i][2] == 0:
            print("flow_acc: flow %d 并未分配P2MP" %i)
        elif flow_count[i][1] > 1 or flow_count[i][2] > 1:
            print("flow_acc: flow %d 被分配了多次" %i)


    node, row, vol = node_P2MP.shape
    if vol == 5 or vol == 12:
        for n in range(topo_num):
            for t in range(Tbox_num):
                scr_band = 0
                des_band = 0
                for p in range(Tbox_P2MP):
                    des_sc = np.zeros((topo_num, 3), dtype=int)
                    P2MP = t * Tbox_P2MP + p
                    for f in node_flow[n][P2MP][0]:
                        if node_P2MP[n][P2MP][2] == 1 or node_P2MP[n][P2MP][2] == 3:
                            scr_band = scr_band + f[3]
                        elif node_P2MP[n][P2MP][2] == 2 or node_P2MP[n][P2MP][2] == 4:
                            des_band = des_band + f[3]
                        des_sc[f[2]][1] = des_sc[f[2]][1] + f[3]
                        des_sc[f[2]][0] = f[4]
                    for j in range(topo_num):
                        if des_sc[j][0] == 12:
                            des_sc[j][2] = math.ceil(des_sc[j][1] / 12.5)
                        else:
                            des_sc[j][2] = math.ceil(des_sc[j][1] / 25)
                    sc = np.sum(des_sc[:, 2])
                    if sc > type_P2MP[node_P2MP[n][P2MP][3] - 1][1]:
                        print("node_flow: node %d P2MP %d 超出容量限制" % (n, P2MP))

                if scr_band > 400 or des_band > 400:
                    print("node_flow: node %d T-box %d 超出容量限制" %(n, t))
    else:
        for n in range(topo_num):
            for t in range(Tbox_num):
                scr_band = 0
                des_band = 0
                for p in range(Tbox_P2MP):
                    P2MP = t * Tbox_P2MP + p
                    for f in node_flow[n][P2MP][0]:
                        if node_P2MP[n][P2MP][2] == 1 or node_P2MP[n][p][2] == 3:
                            scr_band = scr_band + f[3]
                        elif node_P2MP[n][P2MP][2] == 2 or node_P2MP[n][p][2] == 4:
                            des_band = des_band + f[3]

                if scr_band > 400 or des_band > 400:
                    print("node_flow: node %d T-box %d 超出容量限制" % (n, t))

    return flow_count