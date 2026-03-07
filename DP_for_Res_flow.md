# DP_for_Res.py 代码流程（当前实现）

## 为什么这样设计
- **目标不是重建，而是增量恢复**：
  - 本实现假设输入是故障后的资源状态，重点是“尽量复用已有分配”，而不是清空重算。
  - 因此流程里保留了增量写入、局部回滚、失败兜底三件事。
- **先统一输入，避免恢复流无状态位**：
  - `_build_restore_inputs(...)` 先把受影响流转成统一记录，并扩容 `flow_acc / flow_path`。
  - 这样后续恢复成功与失败都能完整落在同一套状态表里。
- **按 `(目的节点, 物理路径)` 分组是为了复用与聚合**：
  - 同目的同路径的流更容易在同一端口/同一树上承载，减少频谱和端口碎片。
  - 这也是先分组再选“最佳分组”的原因。
- **先做单层背包选流，再逐流可行性校验**：
  - 单层背包只负责“在端口剩余接收容量约束下，先挑哪批流”。
  - 真正 SC/FS/链路/leaf 的可行性，由 `_plan_sc_allocation(...)` 逐流确认。
  - 这样把“全局挑选”与“细粒度约束检查”解耦，逻辑更稳。
- **`modu_format_Al(dist, 0)` 的意图**：
  - 先拿该距离下单 SC 的能力档位（路径能力），再结合每条流带宽算 `sc_needed`。
  - 这样路径能力与流量大小分层处理，便于分组阶段统一估算。
- **提交前再检查与选择性回滚**：
  - `_plan_sc_allocation` 成功后仍做 `_fs_rel_segment_empty` 复核，防止提交阶段冲突。
  - 回滚只针对“本轮新启用且未成功放置”的端口，避免伤及既有业务。
- **必须保留顺序兜底**：
  - DP 未放下的流进入 `rest_flows`，再交给 `allocate_flows_sequential(...)`。
  - 这保证“先优化恢复质量，再保证恢复成功率”的策略。

## 1. 整体入口
- 入口函数是 `restore_with_dp(...)`：
  - 先调用 `_build_restore_inputs(...)` 把 `affected_flow` 转成恢复用 `flows_info`；
  - 扩展 `flow_acc / flow_path`，给新恢复流预留记录位；
  - 再调用 `FlexE_P2MP_DP(...)` 做 DP 恢复。

## 2. `_build_restore_inputs`：构造恢复输入
- 对每条受影响流：
  - 用 `k_shortest_path` 计算源宿最短路径距离；
  - 用 `modu_format_Al + sc_effective_cap + sc_num_from_bw_cap` 计算 `sc_cap/sc_num`；
  - 生成记录 `[new_id, src, dst, bw, sc_cap, sc_num, orig_id]`。
- 按新增流数量扩容 `flow_acc` 与 `flow_path`：
  - `flow_acc` 新行默认置位（7:15 = -1）；
  - `flow_path` 新行各列初始化为空列表。

## 3. `FlexE_P2MP_DP`：恢复主流程

### 3.1 预处理
- 由 `_TYPE_INFO` 构造端口类型表 `type_P2MP`。
- 按源节点把 `flows_info` 分类到 `scr_flows[n]`。

### 3.2 内部辅助函数
- `_find_free_fs_block_on_links(link_FS, used_links, block_size)`：
  - 在给定链路集合上找一个长度为 `block_size` 的连续空闲 FS 段起点。
- `_fs_rel_segment_empty(P2MP_FS, u, p, fs_rel_s, fs_rel_e)`：
  - 判断端口相对 FS 区间是否全空，用于提交前再校验。

### 3.3 按源节点恢复
- 外层循环：`for n in range(topo_num)`。
- 初始化：
  - `group_map`：按 `(目的节点, 物理路径)` 聚合流；
  - `rest_flows`：DP 失败后待兜底的流。

### 3.4 分组阶段
- 遍历 `scr_flows[n]` 每条流：
  - 用最短路径得到 `phys_nodes/dist`；
  - 转换出 `used_links`；
  - 计算每条流的 `sc_needed`；
  - 以 `(des, tuple(phys_nodes))` 作为 key 放入 `group_map`。

### 3.5 每个源端口选择“最佳分组”
- 遍历该源节点所有端口 `src_p`：
  - 若所有分组都空，提前结束端口循环；
  - 读取端口类型参数：`sc_limit/fs_size`。
- 对每个候选分组做单层背包评估：
  - 确定可用 `fs_start`（端口已有基址则复用，否则在链路上找空闲块）；
  - 背包容量 = `node_P2MP[n][src_p][4]`（剩余接收容量）；
  - 背包物品 = `[fid, bw]`；
  - 用 `knapsack_DP(cap_bw, items_for_group)` 选流；
  - 评分 = 选中流总带宽，保留得分最高分组为 `best_key`。

### 3.6 对最佳分组逐流调用 `_plan_sc_allocation`
- 若没有可用最佳分组则 `continue` 下一个端口。
- 端口状态处理：
  - `node_P2MP[n][src_p][2] = 1`；
  - 若端口无 base FS，则写入 `best_fs_start`；
  - 若端口原本无流，清空端口记录并 `_clear_p2mp_sc`；有流则增量写入。
- 对 `best_selected` 每条流：
  - 调用 `_plan_sc_allocation(...)` 规划该流 SC/FS（逐流）；
  - 失败 -> 加入 `rest_flows`；
  - 成功后再做 `_fs_rel_segment_empty` 提交前检查；
  - 提交时更新：
    - `node_flow`、`flow_acc`；
    - `P2MP_SC`（hub侧）、`P2MP_FS`；
    - `flow_path`；
    - `_apply_leaf_usage(...)` + leaf 侧 `_apply_p2mp_fs_usage(...)`；
    - `link_FS` 链路 FS 占用。
- 本轮处理过的流（成功或失败）从 `group_map[best_key]["flows"]` 移除。
- 若该端口本轮没有成功放置任何流，且它是本次新启用端口，则回滚该端口状态。

### 3.7 兜底恢复
- 把 `group_map` 剩余流全部放进 `rest_flows`。
- 若 `rest_flows` 非空，调用 `allocate_flows_sequential(...)` 逐条兜底恢复。

### 3.8 返回
- 返回：
  - `node_flow, node_P2MP, flow_acc, link_FS, P2MP_SC, flow_path`

---

## 伪代码（与当前实现一致）

```text
function restore_with_dp(...):
    affected_flows_info, flow_acc, flow_path = _build_restore_inputs(...)
    return FlexE_P2MP_DP(affected_flows_info, ..., flow_acc, flow_path)

function FlexE_P2MP_DP(...):
    构造 type_P2MP
    按源节点分类 flows -> scr_flows

    定义 _find_free_fs_block_on_links(...)
    定义 _fs_rel_segment_empty(...)

    for each source n:
        group_map = {}
        rest_flows = []

        for each flow f in scr_flows[n]:
            path = shortest_path(n, des)
            used_links = links_of(path)
            sc_needed = calc_sc_needed(path, bw)
            group_map[(des, path)].flows append f

        for each src_p:
            if all groups empty: break

            best_key = None
            best_selected = []
            best_fs_start = None
            best_score = -1

            for each group in group_map:
                fs_start = existing_base_or_find_free(...)
                if fs_start invalid: continue

                cap_bw = node_P2MP[n][src_p][4]
                selected = knapsack_DP(cap_bw, items=[fid,bw])
                if selected empty: continue

                score = sum(bw of selected)
                keep best group by score

            if no best group: continue

            activate/reuse port state

            for each flow in best_selected:
                plan = _plan_sc_allocation(...)
                if fail: rest_flows.append(flow); continue
                if fs segment not empty: rest_flows.append(flow); continue
                commit node_flow/flow_acc/P2MP_SC/P2MP_FS/flow_path/link_FS

            remove processed flows from best group
            rollback newly-activated empty port if none placed

        collect remaining group flows into rest_flows
        if rest_flows not empty:
            allocate_flows_sequential(rest_flows)

    return updated resource tables
```
