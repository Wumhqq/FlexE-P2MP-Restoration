#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @IDE     : PyCharm 
# @Project : code 
# @File    : main.py
# @Author  : Wumh
# @Time    : 2024/9/12 22:28
import copy
import math
import numpy as np
from topology import topology
import random
from k_shortest_path import k_shortest_path
from modu_format_Al import modu_format_Al
from Initial_network_modified_125_ts_samefs_v2 import FlexE_P2MP_DP
from test_right import test_right
from ILP import Restoration_ILP
from Heuristic_al_fix import Heuristic_algorithm
from ILP_main import Restoration_ILP
import traceback
import pdb


TS_UNIT = 5  # TS 粒度（Gbps），与 ILP 中 *5 的定义保持一致

def sc_effective_cap(sc_cap: float) -> int:
    """把 SC 容量映射成可用的整数 TS 容量（不允许 2.5 TS 这种分配）。
    例如：12.5Gbps 只能承载 floor(12.5/5)=2 个 TS => 10Gbps。
    """
    return int(math.floor(float(sc_cap) / TS_UNIT) * TS_UNIT)

def sc_num_from_bw_cap(bw: float, sc_cap: float) -> int:
    eff = sc_effective_cap(sc_cap)
    if eff <= 0:
        raise ValueError(f"Invalid SC_cap={sc_cap}")
    return int(math.ceil(float(bw) / eff))

def main():
    flows_sum = 500
    topo_num, topo_matrix, topo_dis, link_num, link_index = topology(1)
    band = np.array([10, 40, 25])
    Tbox_num = 60#均分成3类
    Tbox_P2MP = 1
    type_P2MP = [[1, 1], [2, 4], [3, 16]]

    # 生成流量：用于保存原始记录
    # flows_info: 0.flow序号 1.源节点 2.目的节点 3.带宽 4.逻辑路径 5.物理路径
    affected_flow = []

    while (len(affected_flow) == 0):
        flows_info = []
        band_sum = 0
        i = 0
        while band_sum < flows_sum:
            info = []
            info.append(i)
            info.append(random.randint(0, topo_num - 1))
            info.append(random.randint(0, topo_num - 1))
            while(info[1] == info[2]):
                info[1] = random.randint(0, topo_num - 1)
                info[2] = random.randint(0, topo_num - 1)
            info.append(band[random.randint(0, len(band) - 1)])
            if info[3] == 25:
                info[3] = info[3] * random.randint(1, 8)
            # 寻找scr和des的最短路径，随机中转节点个数和中转节点
            # 注意：k_shortest_path的scr和des是从1开始
            path = k_shortest_path(topo_dis, info[1] + 1, info[2] + 1, 1)
            OEO_num = random.randint(0, len(path[0][0]) - 2)
            OEO_node = random.sample(path[0][0][1:-1], OEO_num)
            sorted_numbers = sorted(OEO_node, key=path[0][0].index)
            if len(sorted_numbers) == 0:
                info.append([path[0][0][0], path[0][0][-1]])
            else:
                logical_path = []
                logical_path.append(path[0][0][0])
                for n in sorted_numbers:
                    logical_path.append(n)
                logical_path.append(path[0][0][-1])
                info.append(logical_path)
            info.append(path[0][0])
            flows_info.append(info)
            band_sum = band_sum + info[3]
            i = i + 1
        flows_num = len(flows_info)

        # 拆解流量：用于初始网络场景分配资源
        # flows_info: 0.flow序号 1.源节点 2.目的节点 3.带宽 4.调制格式 5.SC 6.原flow序号
        # disass_flows_info: 0.flow序号 1.源节点 2.目的节点 3.带宽 4.调制格式 5.SC 6.原flow序号
        disass_flows_info = []
        flow_index = 0
        for f in flows_info:
            for i in range(len(f[4]) - 1):
                temp = []
                j = i + 1
                temp.append(f[0])
                temp.append(f[4][i] - 1)
                temp.append(f[4][j] - 1)
                temp.append(f[3])
                # 根据物理路径计算逻辑路径的长度
                i_index = f[5].index(f[4][i])
                j_index = f[5].index(f[4][j])
                dist = 0
                for k in range(i_index, j_index):
                    a = f[5][k] - 1
                    b = f[5][k + 1] - 1
                    dist = dist + topo_dis[a][b]
                SC_cap, SC_num = modu_format_Al(dist, f[3])
                # 保留真实的 SC 容量（例如 12.5）；但 SC_num 按“整数 TS(=5Gbps) 可分配”口径计算
                SC_num = sc_num_from_bw_cap(f[3], SC_cap)
                temp.append(float(SC_cap))
                temp.append(int(SC_num))
                temp.append(f[0])
                temp[0] = flow_index
                flow_index = flow_index + 1
                disass_flows_info.append(temp)
        disass_flows_info = np.array(disass_flows_info, dtype=object)
        flows_num, _ = np.shape(disass_flows_info)
        try:
            node_flow_DP, node_P2MP_DP, flow_acc_DP, link_FS_DP, P2MP_SC_DP, flow_path_DP = FlexE_P2MP_DP(disass_flows_info, flows_num, topo_num, topo_dis, link_num, link_index, Tbox_num, Tbox_P2MP)
        except Exception:
            traceback.print_exc()
            # pdb.post_mortem()  # 异常现场调试：停在报错行
            raise
        #     # 结果
        # node_flow_DP, node_P2MP_DP, flow_acc_DP, link_FS_DP, P2MP_SC_DP, flow_path_DP = FlexE_P2MP_DP(disass_flows_info, flows_num, topo_num, topo_dis, link_num, link_index, Tbox_num, Tbox_P2MP)

        # test_right
        #test_right(node_P2MP_DP, node_flow_DP, flow_acc_DP, flows_num, topo_num, Tbox_num, Tbox_P2MP, 1)
        #print("FlexE_P2MP_DP: 检查结束")

        # 随机某个节点中断，清空中间节点上的资源使用
        new_flow_acc = copy.deepcopy(flow_acc_DP)
        new_node_flow = copy.deepcopy(node_flow_DP)
        new_node_P2MP = copy.deepcopy(node_P2MP_DP)
        new_link_FS = copy.deepcopy(link_FS_DP)
        new_P2MP_SC = copy.deepcopy(P2MP_SC_DP)
        break_node = random.randint(0, topo_num - 1) # 因为D算法生成的路径是从1开始的

        for f in flows_info:
            if break_node + 1 in f[4]:
                # 源节点和目的节点是中断节点的话，删除所有的资源，且该条流不能被恢复
                if f[4][0] != break_node + 1 and f[4][-1] != break_node + 1:
                    affected_flow.append(f)

                for i in new_flow_acc:
                    if i[6] == f[0]:
                        new_index = i[0]
                        scr = i[1]
                        des = i[2]
                        hub = i[7]
                        leaf = i[8]
                        for index, j in enumerate(new_node_flow[scr][hub][0]):
                            if j[0] == new_index:
                                new_node_flow[scr][hub][0].pop(index)
                                break
                        for index, j in enumerate(new_node_flow[des][leaf][0]):
                            if j[0] == new_index:
                                new_node_flow[des][leaf][0].pop(index)
                                break
                        #new_node_P2MP[des][leaf][4] = new_node_P2MP[des][leaf][4] + i[3]
                        for s in range(i[9], i[10] + 1):
                            for index, ff in enumerate(new_P2MP_SC[scr][hub][s][1]):
                                if ff[0] == new_index:
                                    new_P2MP_SC[scr][hub][s][1].pop(index)
                        for l in flow_path_DP[new_index][1]:
                            new_link_FS[l][i[11] : i[12] + 1] = new_link_FS[l][i[11] : i[12] + 1] - 1

                        i[7] = -1
                        i[8] = -1
                        i[9] = -1
                        i[10] = -1
                        i[11] = -1
                        i[12] = -1
    # 更新一下FlexE Group的剩余容量大小
    for u in range(topo_num):
        for p in range(Tbox_num * Tbox_P2MP):
            for f in new_node_flow[u][p][0]:
                new_node_P2MP[u][p][4] = new_node_P2MP[u][p][4] - f[3]

    # 更新一下new_P2MP_SC的使用量
    new_P2MP_SC_1 = copy.deepcopy(new_P2MP_SC)
    for u in range(topo_num):
        for p in range(Tbox_num * Tbox_P2MP):
            for s in range(16):
                if len(new_P2MP_SC_1[u][p][s][1]) != 0:
                    new_P2MP_SC_1[u][p][s][3] = new_P2MP_SC_1[u][p][s][0] - sum(k[1] for k in new_P2MP_SC_1[u][p][s][1])
                    new_P2MP_SC_1[u][p][s][3] = list(set(new_P2MP_SC_1[u][p][s][3]))
                else:
                    new_P2MP_SC_1[u][p][s][0] = []
                    new_P2MP_SC_1[u][p][s][2] = []


    # A = Restoration_ILP(affected_flow, new_flow_acc, new_node_flow, new_node_P2MP, break_node, Tbox_num, Tbox_P2MP,
    #                    new_P2MP_SC_1, new_link_FS, node_flow_DP, node_P2MP_DP, flow_acc_DP, link_FS_DP, P2MP_SC_DP, flow_path_DP)

    D = Restoration_ILP(affected_flow, new_flow_acc, new_node_flow, new_node_P2MP, break_node, Tbox_num, Tbox_P2MP,
                       new_P2MP_SC_1, new_link_FS, node_flow_DP, node_P2MP_DP, flow_acc_DP, link_FS_DP, P2MP_SC_DP, flow_path_DP)
    if D is None:
        print("Error: Restoration_ILP returned None. Optimization might not have run or returned results.")
    else:
        print("Restoration_ILP completed successfully.")

    B = Heuristic_algorithm(affected_flow, new_flow_acc, new_node_flow, new_node_P2MP, break_node, Tbox_num, Tbox_P2MP,
                        new_P2MP_SC_1, new_link_FS, node_flow_DP, node_P2MP_DP, flow_acc_DP, link_FS_DP, P2MP_SC_DP, flow_path_DP)
    if B is None:
         print("Error: Heuristic_algorithm returned None.")
    else:
        print("Heuristic_algorithm completed successfully.")

if __name__ == "__main__":
    main()
