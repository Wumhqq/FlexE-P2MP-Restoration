# AG-S1F 代码流程说明

本文档说明 [AG_S1F.py](/d:/onedrive备份/论文/FlexE-over-P2MP%20restoration/FlexE-P2MP-Restoration/AG_S1F.py) 的整体执行流程、主要函数职责，以及恢复过程中各个状态表是如何被读取和更新的。

## 1. 文件目标

`AG_S1F.py` 实现的是一个“策略 1 优先”的启发式恢复算法。

它的核心思路是：

1. 先从 DP 基线状态中提取受影响业务的历史资源占用信息。
2. 对每条受影响流构建可用的恢复候选。
3. 优先尝试策略 1：
   - `strict_s1=True`
   - 尽量复用原有端口、原有 SC 范围、原有硬件能力
4. 如果策略 1 失败，再进入策略 2：
   - `strict_s1=False`
   - 允许端口/SC/硬件约束放宽
5. 在真正恢复时，支持两种模式：
   - `singlehop`：直接从 `src -> dst` 做单跳恢复
   - `multihop`：先在逻辑拓扑上找路径，再逐跳恢复

## 2. 主要状态表

算法主要操作以下几类状态：

- `new_flow_acc`
  - 子流级资源记录表
  - 恢复出来的新 hop 会追加到这个表的末尾
- `new_node_flow`
  - 每个节点、每个 P2MP 端口当前承载的子流列表
- `new_node_P2MP`
  - 每个节点、每个 P2MP 端口的状态表
  - 例如是否启用、端口类型、剩余容量、FS base 等
- `new_P2MP_SC`
  - 端口的 SC 级占用详情
- `new_P2MP_FS`
  - 端口的 FS 级占用详情
- `new_link_FS`
  - 链路级绝对 FS 占用
- `new_flow_path`
  - 子流对应的链路路径、物理节点路径等信息

## 3. 顶层函数：`Heuristic_algorithm`

`Heuristic_algorithm(...)` 是这个文件的主入口。

它的大致流程如下。

### 3.1 读取拓扑与参数

函数开始先读取：

- 拓扑信息 `topology(1)`
- 链路索引 `link_index`
- 物理候选数 `K_PHY_CANDIDATES`
- 逻辑候选数 `K_LOGICAL_PATHS`
- 重构惩罚 `RECONFIG_PENALTY`
- 跳数惩罚 `HOP_PENALTY`

同时校验 `restore_mode`：

- `singlehop`
- `multihop`

## 3.2 预构建物理路径池 `phy_pool`

对每一对节点 `(u, v)`：

1. 调用 `k_shortest_path(topo_dis, u+1, v+1, K_PHY_CANDIDATES)`
2. 生成若干条物理候选路径
3. 将路径从 1-based 转成 0-based
4. 写入 `phy_pool[(u, v)] = [{"path": ..., "dist": ...}, ...]`

这个池后面会被：

- `build_virtual_topology(...)`
- `_attempt_restore_flow_singlehop(...)`

共同使用。

## 3.3 构建历史元信息 `flow_metadata_map`

通过 `_build_flow_metadata_map(...)`：

1. 遍历每条受影响流
2. 调用 `extract_old_flow_info(...)`
3. 提取该流在 DP 基线中的历史信息

主要包括：

- 源端 hub 端口编号
- 目的端 leaf 端口编号
- 原始 SC 范围
- 原始 SC 使用量
- 端口容量，用于推断硬件调制

这些信息主要用于：

- `strict_s1=True` 时约束恢复
- `manage_s1_reservation_by_copy(...)` 做首尾端资源预占位

## 3.4 初始化当前路径状态

当前版本里，`Heuristic_algorithm(...)` 会直接接收外部传入的 `new_flow_path`。

这里的 `new_flow_path` 语义是：

- 故障发生后、旧资源已经释放之后的当前路径状态
- 也是算法后续继续恢复时要直接更新的路径状态

也就是说：

- 当前主流程不再在函数内部从 `flow_path_base` 派生出一份新的路径副本
- 而是由外部先准备好“故障后状态”，再作为 `new_flow_path` 传入

这样做的好处是：

- 参数语义更直接
- 与 `main.py` 中“先构造故障后状态，再分别交给不同恢复算法”的调用方式一致

## 3.5 为每条流构建 S1/S2 的虚拟拓扑

对每条受影响流 `f`：

1. 从 `flow_metadata_map` 中取出历史 hub/leaf 容量
2. 用 `_infer_hw_modu_from_cap(...)` 推断源端/宿端硬件调制类型
3. 构建策略 1 的虚拟拓扑：
   - `build_virtual_topology(..., force_strategy1=True, ...)`
4. 构建策略 2 的虚拟拓扑：
   - `build_virtual_topology(..., force_strategy1=False, ...)`
5. 分别存入：
   - `flow_vmap_s1[fid]`
   - `flow_vmap_s2[fid]`

## 3.6 判断流是否先进入 Tier1

通过 `_mode_has_s1_candidate(...)` 判断该流是否值得先尝试策略 1。

### `singlehop` 模式

判断规则是：

- `(src, dst)` 是否直接存在于 `v_phy_s1`

也就是策略 1 下是否存在一个可用的单跳虚拟边。

### `multihop` 模式

判断规则是：

1. 在 `v_adj_s1` 上跑一次 `k_shortest_path`
2. 得到一条逻辑路径候选
3. 再用 `logical_path_is_valid(...)` 检查这条路径是否合法

如果合法，则可以进入 `tier1_flows`。

否则进入 `tier2_flows`。

## 3.7 对 Tier1 做首尾端预占位

调用：

`manage_s1_reservation_by_copy(...)`

它的作用是：

1. 从 DP 基线状态中，把该流首尾两端的 SC 占用复制到新的状态表中
2. 扣减首尾端对应端口的剩余容量
3. 防止后续流抢占本流在策略 1 下需要保留的端口/SC 资源

注意：

- 这里只处理“原始流源端 hub”和“原始流宿端 leaf”
- 不做中间 hop 的真实恢复

## 3.8 逐流尝试 Tier1 恢复

对每条 `tier1_flows` 中的流：

1. 先调用 `manage_s1_reservation_by_copy(..., action="rollback")`
   - 把当前流自己的首尾端预占位先撤掉
   - 避免影响“真实恢复分配”
2. 取出本流的 `v_adj_s1` 和 `v_phy_s1`
3. 调用 `_attempt_restore_flow_by_mode(...)`
   - 按 `restore_mode` 决定用单跳还是多跳
4. 若恢复成功：
   - 用返回的新状态覆盖主状态
   - 放入 `tier1_restored`
5. 若恢复失败：
   - 放入 `tier1_failed`
   - 同时转入 `tier2_flows`

## 3.9 逐流尝试 Tier2 恢复

对每条 `tier2_flows`：

1. 取出 `v_adj_s2` 和 `v_phy_s2`
2. 调用 `_attempt_restore_flow_by_mode(...)`
   - 但此时 `strict_s1=False`
3. 成功则提交新状态
4. 失败则加入 `tier2_failed`

最后：

- `failed_orig = [int(f[0]) for f in tier2_failed]`
- 返回最终资源状态和失败原始流编号列表

## 4. 模式分发：`_attempt_restore_flow_by_mode`

这个函数只是一个简单分发器：

- `singlehop` -> `_attempt_restore_flow_singlehop(...)`
- `multihop` -> `_attempt_restore_flow_multihop(...)`

## 5. 单跳模式：`_attempt_restore_flow_singlehop`

这个模式的思路很直接：

1. 取出当前流的 `src` 和 `dst`
2. 在 `phy_pool[(src, dst)]` 里取出所有直接物理候选路径
3. 逐条尝试：
   - 为当前候选路径创建工作副本
   - 调用 `_assign_one_hop_with_initialnet(...)`
4. 只要有一条直接物理候选成功，就返回成功

特点：

- 不走逻辑拓扑
- 不做中间节点 OEO 分解
- 更像“直接恢复一条端到端物理候选”

## 6. 多跳模式：`_attempt_restore_flow_multihop`

这个模式会先在逻辑拓扑层找路，再逐跳分配。

流程如下：

1. 在 `v_adj` 上调用 `k_shortest_path(...)`，得到若干逻辑路径
2. 对每条逻辑路径：
   - 检查是否合法
   - 转成 0-based 节点序列
   - 为整条路径创建工作副本
3. 对逻辑路径中的每一跳 `(a, b)`：
   - 在 `v_phy[(a, b)]` 中取出该逻辑跳对应的最佳物理候选
   - 调用 `_assign_one_hop_with_initialnet(...)`
4. 如果整条逻辑路径的所有 hop 都成功：
   - 返回该工作副本
5. 若当前逻辑路径失败：
   - 丢弃工作副本，尝试下一条逻辑路径

特点：

- 能处理带 OEO 中转的多跳恢复
- 每一跳都复用统一的单跳资源分配逻辑

## 7. 单跳资源分配核心：`_assign_one_hop_with_initialnet`

这是 AG-S1F 中最关键的资源分配函数之一。

它负责给一个 hop `(a -> b)` 和一条物理路径分配资源。

### 7.1 前置检查

先检查：

- `phy_path0` 是否为空
- 路径长度是否至少为 2
- `link_index[u][v]` 是否有效

同时会得到：

- `used_links`
- `phys_nodes_1b`
- `sc_cap`

### 7.2 `strict_s1=True` 时固定端口与历史 SC 约束

如果是策略 1 严格模式：

1. 判断当前 hop 是否落在原始流源端
   - 若是，则尝试固定 hub 端口
2. 判断当前 hop 是否落在原始流宿端
   - 若是，则尝试固定 leaf 端口
3. 从 `meta` 中取出：
   - 历史 SC 范围
   - 历史每个 SC 的使用量
4. 用这些历史信息约束 `_plan_sc_allocation(...)`

### 7.3 先尝试已启用 hub 端口

遍历 `hub_candidates`：

1. 要求端口状态满足“已启用”
2. 检查剩余容量是否足够
3. 检查 base FS 是否有效
4. 调用 `_plan_sc_allocation(...)` 做 SC/FS 联合规划
5. 如果规划成功，调用 `_commit_plan(...)` 提交

### 7.4 再尝试空闲 hub 端口

如果所有已启用端口都失败：

1. 遍历空闲端口
2. 通过 `_find_free_fs_block(...)` 找可用 FS block
3. 暂时激活该端口并清空其 SC 状态
4. 再次调用 `_plan_sc_allocation(...)`
5. 若规划失败：
   - 回滚临时激活状态
6. 若规划成功：
   - 调用 `_commit_plan(...)` 提交

## 8. 为什么恢复出来的子流要追加，而不是复用旧 `flow_acc` 行

这是当前版本里一个很关键的设计点。

旧版本的问题是：

- 会尝试用 `(orig_fid, a, b)` 去匹配旧 `flow_acc` 中已有的子流
- 但恢复后的 hop 很可能和故障前并不一致
- 所以旧 `flow_acc` 里未必存在对应 `(a, b)` 的子流记录

当前版本改成：

- 恢复成功时才追加新的子流记录
- 由 `_append_restored_subflow(...)` 负责：
  - 给 `new_flow_acc` 追加一行
  - 给 `new_flow_path` 追加一行
  - 返回新的 `subflow_id`

也就是说：

- 故障前旧记录保留其历史意义
- 故障后恢复出来的新 hop 作为“新资源记录”附加在后面

## 9. 提交写回：`_commit_plan`

`_commit_plan(...)` 会把规划结果一次性写回所有资源表。

### 9.1 先创建新的恢复子流

先计算：

- `sc_num = sc_e - sc_s + 1`

再调用：

- `_append_restored_subflow(...)`

得到新的 `sub_id`

### 9.2 更新 `new_node_flow`

把该子流写入：

- `new_node_flow[a][hub_p][0]`
- `new_node_flow[b][leaf_p][0]`

### 9.3 更新 `new_node_P2MP`

主要更新：

- 源端 hub 剩余容量 `[4]`
- 目的端 leaf 剩余容量 `[4]`

并在必要时把端口状态从 0 置为 1。

### 9.4 更新 `new_flow_acc`

将新子流的完整信息写入追加出的那一行：

- 子流 ID
- 源宿节点
- 带宽
- SC 容量
- SC 数量
- 原始流 ID
- hub/leaf 端口
- hub SC 范围
- FS 范围
- leaf SC 范围

### 9.5 更新 `new_P2MP_SC`

对源端和宿端涉及到的每个 SC：

- 追加 `used_list`
- 追加 `cap_list`
- 追加路径信息
- 追加源宿节点与端口标记

### 9.6 更新 `new_P2MP_FS`

通过 `_apply_p2mp_fs_usage(...)`：

- 把 hub 侧和 leaf 侧的 SC 占用映射到对应 FS

### 9.7 更新 `new_flow_path`

将以下信息写入新子流的路径记录：

- 子流 ID
- 使用的链路
- 物理节点路径
- 原始流 ID

### 9.8 更新 `new_link_FS`

对当前 hop 用到的链路和 FS 区间：

- `new_link_FS[l][fs_s_glo:fs_e_glo+1] += 1`

## 10. 辅助函数说明

### `sc_effective_cap(sc_cap)`

把理论 SC 容量压成按 TS 粒度可分配的容量。

### `sc_num_from_bw_cap(bw, sc_cap)`

根据带宽和有效 SC 容量，得到所需 SC 数。

### `logical_path_is_valid(path_1b, break_node_0b)`

判断逻辑路径是否合理，只有满足以下条件时才返回 `True`：

- 路径非空
- 路径长度至少为 2
- 不含断点
- 不含重复节点

### `_infer_hw_modu_from_cap(cap)`

根据容量粗略推断硬件调制类型。

### `_build_flow_metadata_map(...)`

为每条受影响流建立一份历史元信息映射表，供后续查用。

## 11. 当前代码的关键特征

当前 AG-S1F 有几个非常重要的实现特点：

1. `Heuristic_algorithm` 只有两种模式：
   - `singlehop`
   - `multihop`
2. 恢复成功时，新 hop 的资源记录会追加到 `new_flow_acc/new_flow_path` 末尾
3. 策略 1 的首尾端占位由 `manage_s1_reservation_by_copy(...)` 负责
4. 单跳和多跳共用同一套 `_assign_one_hop_with_initialnet(...)` 和 `_commit_plan(...)`
5. 所有“尝试路径”的过程都基于工作副本，避免失败污染全局状态

## 12. 一句话总结

AG-S1F 的执行框架可以概括为：

1. 先用 DP 历史状态构造“恢复约束”和“候选拓扑”
2. 再按 `singlehop` 或 `multihop` 模式逐流尝试恢复
3. 每次真正恢复成功时，向 `new_flow_acc/new_flow_path` 追加新的子流记录
4. 最后统一返回更新后的全量资源状态
