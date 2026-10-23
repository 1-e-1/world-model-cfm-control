> Historical source documentation. See the repository root README and PUBLICATION_NOTES.md for current distribution limits.

# Zbot Active Mapping

2026-09-13: non-learning improvement development has started; the stable classical
entry below remains unchanged. Structured task-nullspace IK restarts and a
120-scene blocks/tools protocol are implemented, **not yet a >90% success result**.
See [改进与扩大随机验证](astronaut_delivery/research/CLASSICAL_EXPANSION_20260913.md).

## Active route (2026-09-12): classical RGB-D, without PPO

The current recommended entry is `python -m astronaut_delivery.rgbd_classical_mission`:
the pre-PPO deterministic dual-arm controller with the measured-map capacity repair
(16M cells, unchanged 2 mm resolution). PPO and FlyWire experiments are inactive,
retained in place for reproducibility; no weights are loaded by this entry.
See [当前主线、启动参数与回退边界](astronaut_delivery/research/ACTIVE_CLASSICAL_ROUTE.md).
Historical results and generic experiment entries below remain unchanged;
their scores are not a new evaluation of this pinned configuration.

## Astronaut dual-arm / D435i

Current research priorities (2026-09-11): improve complete-task feasibility and
RGB-D guidance under symmetric-arm viewpoint constraints, then minimize combined
base reaction. VLM recognition and whole-stack hard realtime are outside the
research objectives. See [收敛后的研究范围与失败修复](astronaut_delivery/research/REVISED_RESEARCH_SCOPE.md).
Procedural maintenance-tool assets: `isaacsim_model/maintenance_tool_catalog.usda`;
new tool-scene generation does not yet establish successful tool delivery.
The optional 16M-cell fine-map capacity repaired reused failure scene 82007:
both objects completed and independent audits passed in
`reports/rgbd_capacity_repair_82007_set_01`. This is development reuse, not an
updated holdout rate; the generic entry remains 12M, while the pinned classical
entry above explicitly selects 16M.

2026-09-11: a local SmolVLM-500M task-selector integration completed one requested
RGB-D-only object transfer with independent collision, placement and recording audits.
It is a constrained, optional upstream interface, not a general semantic grasping system.
See [系统输入输出、VLM 实测与实用边界](astronaut_delivery/research/VLM_INTEGRATION_AND_SYSTEM_BOUNDARY.md).
Entry: `python -m astronaut_delivery.rgbd_vlm_transfer`; evidence:
`reports/rgbd_vlm_transfer_01`, `reports/vlm_integration_summary_01/summary.json`.

Current requirement: **environment collision evidence must come from RGB-D**.
The entries below that use known environment CAD are retained historical baselines,
not substitutes for the current measured-environment task.

Generic measured-only mission entry: `python -m astronaut_delivery.rgbd_mission`.
See [RGB-D 路线与已验证边界](astronaut_delivery/research/RGBD_ONLY_ROUTE.md) and
[方法、数学与实验协议](astronaut_delivery/research/RGBD_METHOD_AND_PROTOCOL.md).
Conditional proofs and their limits are separated in
[条件性数学性质](astronaut_delivery/research/RGBD_CONDITIONAL_PROPERTIES.md).
The nominal RGB-D-only three-object mission passed independent audits in
`reports/rgbd_nominal_multi_mission_08`: 169 segments, 12,933 cross-checked reaction
samples, 0.607/0.857/0.690 mm placement errors and 80 mm withdrawal per object.
The full nonhardware project is **not complete**: randomized trials have finished,
but comparative limitations, unresolved failure audits and whole-stack computation
constraints remain outstanding.
See [复现与当前论文入口](astronaut_delivery/research/RGBD_REPRODUCIBILITY.md) and
[英文仿真论文工作稿](astronaut_delivery/research/PAPER_RGBD_SIMULATION.md).
Current nonhardware acceptance and experiment definitions:
[验收矩阵](astronaut_delivery/research/NONHARDWARE_ACCEPTANCE.md),
[纯 RGB-D 对照协议](astronaut_delivery/research/RGBD_EXPERIMENT_PROTOCOL.md).
The persistent multi-object development entry is `python -m astronaut_delivery.rgbd_mission`.
The first frozen nine-trial random development pilot is complete: audited whole-scene
successes are set 0/3, pose 1/3 and hold 1/3. There are no common successful scenes
for a complete-task set-versus-baseline reaction comparison. Repairs are development
work and are not pooled into that frozen batch.
The repaired development scene 6103 completed both objects in
`reports/rgbd_pilot_repair_03_6103_set` (123 audited segments and 9,196 wrench samples).
This reused-scene repair result is not pooled with the frozen pilot.
The second frozen pilot is also complete: set 1/3, pose 2/3 and hold 1/3;
the only common successful scene shows both torque RMS benefits and adverse peaks.
The latest full-project test suite is `logs/pytest_capacity_tools_full_01.xml` (496 passed,
including `piaoyi_model/tests`, VLM interface checks and tool/map-capacity regressions);
unit tests do not establish randomized mission success.
Post-pilot wall-clock watchdogs and bounded contact-corridor checks are implemented.
`reports/rgbd_refinement_development_01/run_6102_set` completes its one-object scene
with independent geometry, reaction and recorded-clock audits; scene 6101 still fails.
Newly rendered nominal/blackout/RGB-blackout/delay runs all completed one reused
development scene, including actual braking and recovery in each fault case;
4,339 online raw records passed recorded-clock audits. See
[新渲染压力实验](astronaut_delivery/research/RGBD_RENDERED_STRESS_PROTOCOL.md).
Vertical-row IK and Euclidean error envelopes preserve the original safety bounds,
but did not alone resolve scene 6101. A measured destination-dispersion heuristic failed the difficult scene and was
removed from the default. The same six-rescan budget is now allocated per newly
observed object, with same-task resets prohibited. Both reused two-object development
scenes 6101 and 6103 completed under this revision, including recorded-clock audits;
neither used a second-task rescan, so this does not isolate the benefit of budget renewal.
The frozen 24-scene, three-policy holdout in `reports/rgbd_holdout_01` is complete:
set 16/24, pose 16/24, hold 18/24. All failures count. Some mean reaction metrics
improve on common successful scenes, but peaks and time do not uniformly improve.
The new VLM development run is not pooled into this frozen holdout.
See the standalone [当前完整技术路线与数学计算](astronaut_delivery/research/CURRENT_RGBD_TECHNICAL_ROUTE.md).
Hardware is excluded, but simulation work remains.
All startup maps require raw RGB-D, matching scene/calibration identity and an
explicit unchanged-environment assumption. No environment CAD fallback is enabled.

### Historical CAD-assisted delivery baselines

Latest scoped implementation: action-local map maintenance, batched analytic reaction allocation,
bounded stale-view recovery and measured open-surface export.
See [独立完整技术路线与数学运算](astronaut_delivery/COMPLETE_TECHNICAL_ROUTE.md).
Results and retained failures: [本次交付与验收](astronaut_delivery/LOCAL_UPDATE_DELIVERY.md).
`python -m astronaut_delivery.sufficiency_delivery --output reports/new_run --objective wrench --anytime-feasible --map-policy local --async-sensor`
is the opt-in entry. It is not a claim of completed hardware deployment or paper-ready validation.

Research branch (opt-in): [论文研究计划与边界](astronaut_delivery/research/RESEARCH_PLAN.md),
[探索性英文论文草稿](astronaut_delivery/research/PAPER_DRAFT.md).
Visual-set allocation is experimental; it does not replace the validated default controller.

Visual-sufficiency claim audit, equations, time-aligned dashboard and randomized-scene evaluation:
[视觉充分性与双臂反作用评估](astronaut_delivery/VISUAL_SUFFICIENCY_EVALUATION.md).

Complete cooperative delivery entry: `python -m astronaut_delivery.cooperative --output reports/my_run --parts 3 --async-sensor`.
See [完整协同项目、运行命令和验收边界](astronaut_delivery/COOPERATIVE_PROJECT.md).
All three parts were delivered in the kinematic benchmark; streaming compute budget passed in the final async run. Hardware/hard-real-time/unknown-map-only safety are not certified.

Online dual-arm / low-reaction prototype: `isaacsim_model/astronaut_delivery_online.usd`.
See [online coordination implementation and measured limits](astronaut_delivery/ONLINE_COORDINATION.md).
The short approach experiment passes; real-time deadlines and full online delivery are **not yet verified**.

Latest camera-specific workcell: `isaacsim_model/astronaut_delivery_d435i.usd`.
Implementation, nominal-vs-hardware calibration boundaries, commands and validation:
[D435i 左臂适配说明](astronaut_delivery/D435I.md).
The original assembly and verified legacy delivery artifacts are preserved.

统一工程根目录：`D:\\work_space\\lunwen2`。不再使用外层 `Codex/` 子工程。

## 保留的三条路线

| 路线 | 当前架构 / 入口 | 保留模型 |
| --- | --- | --- |
| 流匹配 CFM | `codex_zbot/cfm/`，SecondOrderCFM | `checkpoints/formal_cfm_v1/best.pt` |
| Drifting | `piaoyi_model/model_v3.py`，V7 一阶时空架构 | `checkpoints/drifting_v7/best.pt` |
| 无神经网络 | `classical_fusion/planning_map.py`，Planning v3 | 无权重；4 mm 鲁棒 TSDF + NBV + 保守碰撞查询 |

`model_v3.py` 和 `*_v4.py` 是仍被 V7 使用的核心/控制接口名称，不是需要再次删除的旧版本。
`codex_zbot` 保留为 Python 包名以保持导入和模型兼容，不再代表额外工程层级。

共享组件在 `codex_zbot/perception/`、`tasks/`、`data/`、`utils/`。
机器人资产 `zbot_8s_new.usd` 保留在根目录。原始采集数据保留在 `data/`；
数据中的 RGB/深度是训练输入，不属于已删除的临时展示图片。

## 环境与测试

从工程根目录运行：

```powershell
$PY = 'D:\\conda_envs\\env_isaaclab\\python.exe'
& $PY -B -m pytest -q -p no:cacheprovider
& $PY -B scripts/verify_project_layout.py
```

IsaacLab 路径：`D:\\Isaac_lab`。涉及原生仿真的脚本继续使用现有 AppLauncher / DirectRLEnv 工作流。
测试和模型兼容检查可在 CPU 运行，不启动采集或占用其他工程的 GPU 训练。

## CFM

架构、损失、奖励和训练阶段见 [ARCHITECTURE.md](ARCHITECTURE.md)。
保留正式模型配套配置 `configs/formal_v1.json`。小型 smoke 与 ablation 配置是同一架构的测试，不是淘汰版本。

```powershell
& $PY -B scripts/evaluate_cfm.py --config configs/formal_v1.json --checkpoint checkpoints/formal_cfm_v1/best.pt --data_dir data/sequences_v1 --split test
```

辅助权重：`checkpoints/formal_world_v1/best.pt`、`checkpoints/formal_ppo_v1/best.pt`。
保留权重不意味着对新随机形状数据完成重新训练。

## Drifting V7

说明见 [piaoyi_model/README.md](piaoyi_model/README.md)。
16 帧、2048 粒子、8192 观测点、一阶漂移目标 12 次训练更新；单个样本推理 1 NFE。

```powershell
& $PY -B -m piaoyi_model.evaluate_v3 --config piaoyi_model/configs/occupancy_drifting_v7_partial_visibility.json --checkpoint checkpoints/drifting_v7/best.pt --data_dir data/sequences_v7_partial_visibility_formal --split test --device cpu
```

`checkpoints/drifting_controller/` 中的 PPO / World Model 是最近保留的配套控制权重，
来自较早的 V6 控制训练，**没有在 V7 几何输出上重新验证闭环**。不能把它们称作已训练的 V7 完整系统。

## 无神经网络 Planning v3

说明与验证边界见 [classical_fusion/PLANNING_V3.md](classical_fusion/PLANNING_V3.md)。
当前只保留 Planning v3 及其共享 TSDF 核心；删除传统融合 v1 的 PPO/World Model 分支。
旧基线仅保留指标 JSON 用作历史对照，不保留实现。

```powershell
& $PY -B -m classical_fusion.active_tsdf_experiment --planning --refine_calibration --start_episode 60 --episodes 20 --view_budget 8 --frames_per_view 5 --modes tsdf_active --output_dir classical_fusion/outputs/planning_v3
```

原生混合形状采集样例从旧展示目录移入 `data/planning_shapes_512_full_depth/`。
相应相机参数：`configs/camera_512_f110_integer_pixels.json`。
已有指标在 `reports/planning_v3/`，PNG/GIF 和临时网格展示文件已删除。
无神经网络路线仍使用 PyTorch 进行数值运算，不加载神经网络模型。

## 清理与安全边界

- `maintenance/cleanup_plan.json`、`cleanup_journal.json`：明确的迁移 / 删除记录。
- `maintenance/retained_weights.json`：保留权重的 SHA-256，可核对搬迁前后内容。
- `maintenance/pre_cleanup_source_20260906.zip`：清理前源码、配置和小型指标记录的恢复备份；不在 Python 导入路径中。
- 备份不包含旧权重、原始数据或展示媒体。淘汰权重、媒体及缓存删除后无法通过这个备份恢复。

当前建图仍存在遮挡和未知区域，不能将网格补洞当作自由空间证据。
真实机械臂需要独立完成全连杆碰撞、自碰撞、臂间碰撞、标定、误差预算、同步和急停验证。
