#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import math
import numpy as np

from k_shortest_path import k_shortest_path
from Path_Link import get_links_from_path
from modu_format_Al import modu_format_Al
from Initial_res import initialize_resource_tables
from Initial_net import allocate_flows_sequential, sc_effective_cap, sc_num_from_bw_cap, TS_UNIT, _clear_p2mp_sc, _apply_p2mp_fs_usage, _plan_sc_allocation, _apply_leaf_usage
from knapsack_DP import knapsack_DP
from SC_FS import sc_fs

# 基于故障后资源状态的 DP 恢复入口（源节点维度 + 连续FS块 + TS背包）
def FlexE_P2MP_DP(flows_info, flows_num, topo_num, topo_dis, link_num, link_index, Tbox_num, Tbox_P2MP, init_resources=None, *, restore_mode=False):
    # 初始化资源表：空表 or 故障后状态
    if init_resources is None:
        if restore_mode:
            raise ValueError("restore_mode=True 时必须传入故障后的资源状态 init_resources")
        # 初始分配：从空资源状态初始化
        (type_P2MP, _p2mp_total, _t1, _t2, link_FS, flows_info, scr_flows,
         node_P2MP, node_flow, flow_acc, P2MP_SC, P2MP_FS, flow_path) = initialize_resource_tables(
            flows_info, flows_num, topo_num, link_num, Tbox_num, Tbox_P2MP
        )
    else:
        # 恢复模式：使用故障后传入的资源表继续分配
        (type_P2MP, _p2mp_total, _t1, _t2, link_FS, flows_info, scr_flows,
         node_P2MP, node_flow, flow_acc, P2MP_SC, P2MP_FS, flow_path) = init_resources

    # 在路径的所有链路上搜索可用的连续 FS 块
    def _find_free_fs_block_on_links(link_FS, used_links, block_size):
        if not used_links:
            combined_usage = link_FS.sum(axis=0) * 0
        else:
            combined_usage = link_FS[used_links].sum(axis=0)
        for i0 in range(len(combined_usage) - int(block_size) + 1):
            if np.all(combined_usage[i0:i0 + int(block_size)] == 0):
                return int(i0)
        return None

    # 判断该端口内某段相对 FS 是否完全空闲
    def _fs_rel_segment_empty(P2MP_FS, u, p, fs_rel_s, fs_rel_e):
        for fs_rel in range(int(fs_rel_s), int(fs_rel_e) + 1):
            try:
                used_list = P2MP_FS[u][p][int(fs_rel)][1]
                path_list = P2MP_FS[u][p][int(fs_rel)][2]
            except Exception:
                return False
            if used_list or path_list:
                return False
        return True

    # 返回该端口可用的最长连续 SC 段长度（其覆盖的 FS 段必须全空）
    def _max_free_sc_run(P2MP_FS, u, p, p_type, sc_limit):
        max_run = 0
        cur_run = 0
        for sc_idx in range(int(sc_limit)):
            fs_rel_s = int(sc_fs(int(p_type), int(sc_idx), 1))
            fs_rel_e = int(sc_fs(int(p_type), int(sc_idx), 2))
            if _fs_rel_segment_empty(P2MP_FS, u, p, fs_rel_s, fs_rel_e):
                cur_run += 1
                if cur_run > max_run:
                    max_run = cur_run
            else:
                cur_run = 0
        return int(max_run)

    # 按源节点逐个恢复
    for n in range(topo_num):
        # 以源节点为单位进行恢复
        # 1. 将待恢复流按 (目的节点, 路径) 分组
        B_flows_info = copy.deepcopy(scr_flows[n])
        
        # 预计算每条流的 SC 需求 (基于路径距离)
        # 并在分组时直接存储
        group_map = {}
        
        # 暂存无法通过 P2MP 恢复的流，稍后尝试逐条恢复
        rest_flows = []
        
        # 初始分组
        # 基于最短物理路径计算 SC 需求并形成分组
        for f in B_flows_info:
            des = int(f[2])
            bw = float(f[3])
            # 重新计算路径 (K=1)
            path = k_shortest_path(topo_dis, n + 1, des + 1, 1)
            phys_nodes = path[0][0]
            dist = float(path[0][1])
            used_links = get_links_from_path(phys_nodes, link_index)
            used_links = [int(x) - 1 for x in used_links]
            
            # 计算 SC 需求
            sc_cap_raw, _ = modu_format_Al(dist, 0)
            sc_eff = sc_effective_cap(sc_cap_raw)
            sc_needed = sc_num_from_bw_cap(bw, sc_eff)
            
            key = (des, tuple(phys_nodes))
            if key not in group_map:
                group_map[key] = {
                    "flows": [],
                    "phys_nodes": phys_nodes,
                    "used_links": used_links,
                    "sc_eff": sc_eff,  # 该路径下的单 SC 有效容量
                    "sc_cap_raw": float(sc_cap_raw)
                }
            # 存入流信息，增加 sc_needed
            # f 结构: [id, src, dst, bw, sc_cap, sc_num, orig_id]
            # 我们需要构造 knapsack item: [id, sc_needed, bw, ...]
            # 这里直接把 f 和 sc_needed 打包
            group_map[key]["flows"].append({
                "original": f,
                "sc_needed": sc_needed,
                "bw": bw
            })

        # 在该源节点的所有端口上尝试恢复
        for src_p in range(_p2mp_total):
            if not any(g["flows"] for g in group_map.values()):
                break
            if int(node_P2MP[n][src_p][2]) != 0 and len(node_flow[n][src_p][0]) > 0:
                continue
            t = int(node_P2MP[n][src_p][3])
            spec = type_P2MP[t - 1]
            sc_limit = int(spec[1])
            fs_size = int(spec[2])
            best_key = None
            best_selected = []
            best_fs_start = None
            best_score = -1
            for key, group in group_map.items():
                if not group["flows"]:
                    continue
                used_links = group["used_links"]
                base_fs = int(node_P2MP[n][src_p][5])
                if base_fs >= 0:
                    fs_start = base_fs
                else:
                    fs_start = _find_free_fs_block_on_links(link_FS, used_links, fs_size)
                    if fs_start is None:
                        continue

                ts_per_sc = int(float(group["sc_eff"]) / float(TS_UNIT))
                if ts_per_sc <= 0:
                    continue
                free_sc_run = _max_free_sc_run(P2MP_FS, n, src_p, t, sc_limit)
                if free_sc_run <= 0:
                    continue
                ts_limit = int(free_sc_run) * int(ts_per_sc)

                items_for_ks = []
                ts_map = {}
                for flow_obj in group["flows"]:
                    fid = int(flow_obj["original"][0])
                    ts_needed = int(flow_obj["sc_needed"]) * int(ts_per_sc)
                    items_for_ks.append([fid, int(ts_needed)])
                    ts_map[fid] = int(ts_needed)

                if not items_for_ks:
                    continue

                _, _, _, final_items = knapsack_DP(int(ts_limit), items_for_ks)
                if not final_items:
                    continue

                selected_ids = set(int(x) for x in final_items)
                selected_flow_objs = [fo for fo in group["flows"] if int(fo["original"][0]) in selected_ids]
                score = sum(ts_map.get(int(fo["original"][0]), 0) for fo in selected_flow_objs)

                if score > best_score:
                    best_score = int(score)
                    best_key = key
                    best_selected = selected_flow_objs
                    best_fs_start = int(fs_start)

            if best_key is None or not best_selected:
                continue

            original_base = int(node_P2MP[n][src_p][5])
            node_P2MP[n][src_p][2] = 1
            node_P2MP[n][src_p][5] = int(best_fs_start)
            node_flow[n][src_p][0].clear()
            _clear_p2mp_sc(P2MP_SC, n, src_p)

            des = int(best_key[0])
            phys_nodes = group_map[best_key]["phys_nodes"]
            used_links = group_map[best_key]["used_links"]
            sc_cap_raw = float(group_map[best_key]["sc_cap_raw"])

            placed_any = False
            processed_ids = set()

            for f_obj in best_selected:
                f = f_obj["original"]
                fid = int(f[0])
                bw = float(f[3])
                plan = _plan_sc_allocation(
                    n, src_p, t, int(best_fs_start), des,
                    phys_nodes, bw, sc_cap_raw, node_flow[n][src_p][0], flow_acc, P2MP_SC,
                    node_P2MP, node_flow, link_FS, P2MP_FS, used_links, flow_path
                )
                processed_ids.add(fid)
                if plan is None:
                    rest_flows.append(f)
                    continue

                sc_s, sc_e, usage, fs_s_glo, fs_e_glo, leaf_p, leaf_sc_s, leaf_sc_e, leaf_usage = plan
                fs_rel_s = int(fs_s_glo) - int(best_fs_start)
                fs_rel_e = int(fs_e_glo) - int(best_fs_start)
                if not _fs_rel_segment_empty(P2MP_FS, n, src_p, fs_rel_s, fs_rel_e):
                    rest_flows.append(f)
                    continue

                node_flow[n][src_p][0].append(f)
                node_flow[des][leaf_p][0].append(f)

                flow_acc[fid][7] = int(src_p)
                flow_acc[fid][8] = int(leaf_p)
                flow_acc[fid][9] = int(sc_s)
                flow_acc[fid][10] = int(sc_e)
                flow_acc[fid][11] = int(fs_s_glo)
                flow_acc[fid][12] = int(fs_e_glo)
                flow_acc[fid][13] = int(leaf_sc_s)
                flow_acc[fid][14] = int(leaf_sc_e)

                for sc, used_amt in usage.items():
                    P2MP_SC[n][src_p][int(sc)][1].append([fid, float(used_amt), f[6]])
                    P2MP_SC[n][src_p][int(sc)][0].append(float(sc_cap_raw))
                    P2MP_SC[n][src_p][int(sc)][2].append(phys_nodes)
                    P2MP_SC[n][src_p][int(sc)][4].append(int(des))
                    P2MP_SC[n][src_p][int(sc)][5].append(int(n))
                    P2MP_SC[n][src_p][int(sc)][6].append(int(src_p))
                    P2MP_SC[n][src_p][int(sc)][7].append(int(des))
                    P2MP_SC[n][src_p][int(sc)][8].append(int(leaf_p))
                    _apply_p2mp_fs_usage(
                        P2MP_FS, n, src_p, int(t), int(sc),
                        fid, float(used_amt), phys_nodes, des, f[6],
                        n, src_p, des, leaf_p
                    )

                flow_path[fid][0].append(fid)
                flow_path[fid][1].extend(used_links)
                flow_path[fid][2].extend(phys_nodes)
                flow_path[fid][3].append(f[6])

                _apply_leaf_usage(
                    P2MP_SC, des, leaf_p,
                    int(leaf_sc_s), int(leaf_sc_e),
                    leaf_usage, float(sc_cap_raw),
                    fid, phys_nodes, des, f[6],
                    n, src_p, des, leaf_p,
                )
                leaf_type = int(node_P2MP[des][leaf_p][3])
                for sc, used_amt in leaf_usage.items():
                    _apply_p2mp_fs_usage(
                        P2MP_FS, des, leaf_p, int(leaf_type), int(sc),
                        fid, float(used_amt), phys_nodes, des, f[6],
                        n, src_p, des, leaf_p
                    )

                for l in set(used_links):
                    link_FS[int(l)][int(fs_s_glo):int(fs_e_glo) + 1] += 1

                placed_any = True

            group_map[best_key]["flows"] = [
                fo for fo in group_map[best_key]["flows"]
                if int(fo["original"][0]) not in processed_ids
            ]

            if not placed_any:
                node_P2MP[n][src_p][2] = 0
                if original_base < 0:
                    node_P2MP[n][src_p][5] = -1
                node_flow[n][src_p][0].clear()
                _clear_p2mp_sc(P2MP_SC, n, src_p)

        for group in group_map.values():
            for f_obj in group["flows"]:
                rest_flows.append(f_obj["original"])

        if rest_flows:
            node_flow, node_P2MP, flow_acc, link_FS, P2MP_SC, P2MP_FS, flow_path, _ = allocate_flows_sequential(
                flows_info=np.array(rest_flows, dtype=object),
                topo_num=topo_num,
                topo_dis=topo_dis,
                link_num=link_num,
                link_index=link_index,
                Tbox_num=Tbox_num,
                Tbox_P2MP=Tbox_P2MP,
                node_flow=node_flow,
                node_P2MP=node_P2MP,
                link_FS=link_FS,
                P2MP_SC=P2MP_SC,
                P2MP_FS=P2MP_FS,
                flow_acc=flow_acc,
                flow_path=flow_path,
                stop_on_fail=False,
            )

    return node_flow, node_P2MP, flow_acc, link_FS, P2MP_SC, flow_path
