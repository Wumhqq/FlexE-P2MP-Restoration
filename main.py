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
from DP_for_Res import FlexE_P2MP_DP
from test_right import test_right
from ILP import Restoration_ILP
from AG_S1F import Heuristic_algorithm
from ILP_main import Restoration_ILP
import traceback
import pdb

from Initial_res import initialize_resource_tables

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

def _disassemble_flows(flows_info: list, topo_dis: np.ndarray) -> np.ndarray:
    disass_flows_info = []
    flow_index = 0
    for f in flows_info:
        # 按逻辑路径的相邻节点拆分成子流
        for i in range(len(f[4]) - 1):
            temp = []
            j = i + 1
            # 子流基础信息：源、宿、带宽
            temp.append(f[0])
            temp.append(f[4][i] - 1)
            temp.append(f[4][j] - 1)
            temp.append(f[3])
            # 根据物理路径中两点的区间计算逻辑段距离
            i_index = f[5].index(f[4][i])
            j_index = f[5].index(f[4][j])
            dist = 0
            for k in range(i_index, j_index):
                a = f[5][k] - 1
                b = f[5][k + 1] - 1
                dist = dist + topo_dis[a][b]
            # 计算调制对应的 SC 容量与所需 SC 数
            SC_cap_raw, SC_num = modu_format_Al(dist, f[3])
            SC_cap = sc_effective_cap(float(SC_cap_raw))
            SC_num = sc_num_from_bw_cap(f[3], SC_cap)
            temp.append(float(SC_cap))
            temp.append(int(SC_num))
            # 记录原始流编号，并重编号子流ID
            temp.append(f[0])
            temp[0] = flow_index
            flow_index = flow_index + 1
            disass_flows_info.append(temp)
    return np.array(disass_flows_info, dtype=object)

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
            # 生成流ID与随机源/宿节点
            info.append(i)
            info.append(random.randint(0, topo_num - 1))
            info.append(random.randint(0, topo_num - 1))
            # 源宿不能相同
            while(info[1] == info[2]):
                info[1] = random.randint(0, topo_num - 1)
                info[2] = random.randint(0, topo_num - 1)
            # 随机带宽
            info.append(band[random.randint(0, len(band) - 1)])
            if info[3] == 25:
                # 25G 带宽按随机倍数扩展
                info[3] = info[3] * random.randint(1, 8)
            # 寻找scr和des的最短路径，随机选择若干中转节点作为逻辑路径
            # 注意：k_shortest_path的scr和des是从1开始
            path = k_shortest_path(topo_dis, info[1] + 1, info[2] + 1, 1)
            OEO_num = random.randint(0, len(path[0][0]) - 2)
            OEO_node = random.sample(path[0][0][1:-1], OEO_num)
            sorted_numbers = sorted(OEO_node, key=path[0][0].index)
            # 组装逻辑路径（1-based节点序列）
            if len(sorted_numbers) == 0:
                info.append([path[0][0][0], path[0][0][-1]])
            else:
                logical_path = []
                logical_path.append(path[0][0][0])
                for n in sorted_numbers:
                    logical_path.append(n)
                logical_path.append(path[0][0][-1])
                info.append(logical_path)
            # 记录完整物理路径（1-based节点序列）
            info.append(path[0][0])
            flows_info.append(info)
            # 更新累计带宽并递增流ID
            band_sum = band_sum + info[3]
            i = i + 1
        flows_num = len(flows_info)

        from Initial_net import FlexE_P2MP_Sequential, restore_flows_sequential
        disass_flows_info = _disassemble_flows(flows_info, topo_dis)
        flows_num, _ = np.shape(disass_flows_info)
        (type_P2MP_init, p2mp_total_init, t1_init, t2_init, link_FS_init, flows_info_init, scr_flows_init,
         node_P2MP_init, node_flow_init, flow_acc_init, P2MP_SC_init, P2MP_FS_init, flow_path_init) = initialize_resource_tables(
            disass_flows_info, flows_num, topo_num, link_num, Tbox_num, Tbox_P2MP
        )

        try:
            node_flow, node_P2MP, flow_acc, link_FS, P2MP_SC, P2MP_FS, flow_path = FlexE_P2MP_Sequential(
                flows_info_init,
                flows_num,
                topo_num,
                topo_dis,
                link_num,
                link_index,
                Tbox_num,
                Tbox_P2MP,
                node_flow_init,
                node_P2MP_init,
                link_FS_init,
                P2MP_SC_init,
                P2MP_FS_init,
                flow_acc_init,
                flow_path_init,
                stop_on_fail=True
            )
        except Exception:
            traceback.print_exc()
            raise


        # 随机某个节点中断，清空中间节点上的资源使用
        new_flow_acc = copy.deepcopy(flow_acc)
        new_node_flow = copy.deepcopy(node_flow)
        new_node_P2MP = copy.deepcopy(node_P2MP)
        new_link_FS = copy.deepcopy(link_FS)
        new_P2MP_SC = copy.deepcopy(P2MP_SC)
        new_P2MP_FS = copy.deepcopy(P2MP_FS)
        new_flow_path = copy.deepcopy(flow_path)
        # 随机选择一个中断节点（内部路径为1-based）
        break_node = random.randint(0, topo_num - 1) 

        for f in flows_info:
            if break_node + 1 in f[4]:
                # 源节点和目的节点是中断节点的话，删除所有的资源，且该条流不能被恢复
                if f[4][0] != break_node + 1 and f[4][-1] != break_node + 1:
                    affected_flow.append(f)

                for i in new_flow_acc:
                    if i[6] == f[0]:
                        # 释放该子流在各类资源结构中的记录
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
                        # leaf 的剩余容量由后续统一重算
                        for s in range(i[9], i[10] + 1):
                            cell = new_P2MP_SC[scr][hub][s]
                            for idx in range(len(cell[1]) - 1, -1, -1):
                                if int(cell[1][idx][0]) == new_index:
                                    for col in (0, 1, 2, 4, 5, 6, 7, 8):
                                        if idx < len(cell[col]):
                                            cell[col].pop(idx)
                        if i[13] != -1 and i[14] != -1:
                            for s in range(i[13], i[14] + 1):
                                cell = new_P2MP_SC[des][leaf][s]
                                for idx in range(len(cell[1]) - 1, -1, -1):
                                    if int(cell[1][idx][0]) == new_index:
                                        for col in (0, 1, 2, 4, 5, 6, 7, 8):
                                            if idx < len(cell[col]):
                                                cell[col].pop(idx)
                        for l in new_flow_path[new_index][1]:
                            new_link_FS[l][i[11] : i[12] + 1] = new_link_FS[l][i[11] : i[12] + 1] - 1
                        # 同步清理 P2MP_FS 的使用记录（用绝对FS映射相对FS）
                        hub_base = int(new_node_P2MP[scr][hub][5])
                        leaf_base = int(new_node_P2MP[des][leaf][5])
                        if hub_base >= 0:
                            for fs_abs in range(i[11], i[12] + 1):
                                fs_rel = int(fs_abs - hub_base)
                                if 0 <= fs_rel < len(new_P2MP_FS[scr][hub]):
                                    cell = new_P2MP_FS[scr][hub][fs_rel]
                                    for idx in range(len(cell[1]) - 1, -1, -1):
                                        if int(cell[1][idx][0]) == new_index:
                                            for col in (1, 2, 4, 5, 6, 7, 8):
                                                if idx < len(cell[col]):
                                                    cell[col].pop(idx)
                        if leaf_base >= 0:
                            for fs_abs in range(i[11], i[12] + 1):
                                fs_rel = int(fs_abs - leaf_base)
                                if 0 <= fs_rel < len(new_P2MP_FS[des][leaf]):
                                    cell = new_P2MP_FS[des][leaf][fs_rel]
                                    for idx in range(len(cell[1]) - 1, -1, -1):
                                        if int(cell[1][idx][0]) == new_index:
                                            for col in (1, 2, 4, 5, 6, 7, 8):
                                                if idx < len(cell[col]):
                                                    cell[col].pop(idx)

                        i[7] = -1
                        i[8] = -1
                        i[9] = -1
                        i[10] = -1
                        i[11] = -1
                        i[12] = -1
                        i[13] = -1
                        i[14] = -1
                        # 清空该子流的路径/链路记录
                        new_flow_path[new_index][0] = []
                        new_flow_path[new_index][1] = []
                        new_flow_path[new_index][2] = []
                        new_flow_path[new_index][3] = []
    # 更新一下FlexE Group的剩余容量大小
    for u in range(topo_num):
        for p in range(Tbox_num * Tbox_P2MP):
            # 用当前端口上承载的流重新扣减剩余容量
            for f in new_node_flow[u][p][0]:
                new_node_P2MP[u][p][4] = new_node_P2MP[u][p][4] - f[3]

    # 更新一下new_P2MP_SC的使用量
    new_P2MP_SC_1 = copy.deepcopy(new_P2MP_SC)
    for u in range(topo_num):
        for p in range(Tbox_num * Tbox_P2MP):
            for s in range(16):
                if len(new_P2MP_SC_1[u][p][s][1]) != 0:
                    # 根据已占用带宽更新SC剩余容量
                    # Fix: index 0 is a list of capacities, take the first one. Index 3 expects a list of [remaining_cap].
                    cap_val = new_P2MP_SC_1[u][p][s][0][0]
                    used_val = sum(k[1] for k in new_P2MP_SC_1[u][p][s][1])
                    new_P2MP_SC_1[u][p][s][3] = [cap_val - used_val]
                else:
                    # 没有使用记录则清空容量与路径
                    new_P2MP_SC_1[u][p][s][0] = []
                    new_P2MP_SC_1[u][p][s][2] = []

    # 为未设置 base_fs 且当前无流的 P2MP 选择全路径可用的 FS 段，找不到则随机生成
    for u in range(topo_num):
        for p in range(Tbox_num * Tbox_P2MP):
            base_fs = int(new_node_P2MP[u][p][5])
            if base_fs >= 0:
                continue
            if len(new_node_flow[u][p][0]) != 0:
                continue
            p_type = int(new_node_P2MP[u][p][3])
            block_size = int(type_P2MP[int(p_type) - 1][2])
            base_fs = None
            combined_usage = new_link_FS.sum(axis=0)
            for i0 in range(len(combined_usage) - block_size + 1):
                if np.all(combined_usage[i0:i0 + block_size] == 0):
                    base_fs = int(i0)
                    break
            if base_fs is None:
                max_start = max(0, new_link_FS.shape[1] - block_size)
                base_fs = int(random.randint(0, max_start))
            new_node_P2MP[u][p][5] = int(base_fs)

    node_flow_s1f, node_P2MP_s1f, flow_acc_s1f, link_FS_s1f, P2MP_SC_s1f, P2MP_FS_s1f, flow_path_s1f = Heuristic_algorithm(
        affected_flow,
        new_flow_acc,
        new_node_flow,
        new_node_P2MP,
        break_node,
        Tbox_num,
        Tbox_P2MP,
        new_P2MP_SC_1,
        new_link_FS,
        new_P2MP_FS,
        node_flow,
        node_P2MP,
        flow_acc,
        link_FS,
        P2MP_SC,
        P2MP_FS,
        flow_path,
    )


    node_flow_r, node_P2MP_r, flow_acc_r, link_FS_r, P2MP_SC_r, P2MP_FS_r, flow_path_r, failed_restore = restore_flows_sequential(
        affected_flow,
        topo_num,
        topo_dis,
        link_num,
        link_index,
        Tbox_num,
        Tbox_P2MP,
        new_node_flow,
        new_node_P2MP,
        new_link_FS,
        new_P2MP_SC_1,
        new_P2MP_FS,
        new_flow_acc,
        new_flow_path,
        break_node,
    )
    if failed_restore:
        print(f"Restoration_Sequential failed for orig_flow_ids={failed_restore}")
    else:
        print("Restoration_Sequential completed successfully.")

    scr_flows_dp = [[] for _ in range(topo_num)]
    for f in disass_flows_info:
        scr_flows_dp[int(f[1])].append(f)
    init_resources_dp = (
        type_P2MP_init, p2mp_total_init, t1_init, t2_init, new_link_FS, disass_flows_info, scr_flows_dp,
        new_node_P2MP, new_node_flow, new_flow_acc, new_P2MP_SC_1, new_P2MP_FS, new_flow_path
    )
    node_flow_dp, node_P2MP_dp, flow_acc_dp, link_FS_dp, P2MP_SC_dp, flow_path_dp = FlexE_P2MP_DP(
        disass_flows_info,
        flows_num,
        topo_num,
        topo_dis,
        link_num,
        link_index,
        Tbox_num,
        Tbox_P2MP,
        init_resources=init_resources_dp,
        restore_mode=True,
    )



if __name__ == "__main__":
    main()
