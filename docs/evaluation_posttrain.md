# RoboMME 单帧后训练：可复用评测

服务器代码根目录：`/mnt/cpfs/heping/SimpleMemVLA`。

统一入口：`scripts/eval_posttrain.sh`。协议与默认 5k 断点：`configs/eval/posttrain_train40_5k.yaml`。固定 episode 清单：`configs/eval/robomme_train40_episode_ids.json`。入口按 openloop → action_report → success → comparison 串行运行；两模型分别占用一张 GPU。训练/模型代码没有因本次整理而改变。

## 手动启动（全新输出目录）

在服务器执行：

```bash
cd /mnt/cpfs/heping/SimpleMemVLA
CFG=/mnt/cpfs/heping/SimpleMemVLA/configs/eval/posttrain_train40_5k.yaml
OUT=/mnt/cpfs/heping/SimpleMemVLA-data/output/fullpretrain_sf_posttrain_20261025_v1/manual_eval_5k_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
nohup bash scripts/eval_posttrain.sh --config "$CFG" --experiment E1 --gpu 6 --output "$OUT/E1" > "$OUT/E1_launcher.log" 2>&1 &
nohup bash scripts/eval_posttrain.sh --config "$CFG" --experiment E2 --gpu 7 --output "$OUT/E2" > "$OUT/E2_launcher.log" 2>&1 &
```

已有这轮评测运行时无需重复启动。只重跑开环或成功率：加 `--stages openloop` 或 `--stages success`，使用新的输出目录。已有目录的后续阶段/失败恢复使用 `--resume`；已通过校验的阶段会复用。单独刷新报告：`--stages report --resume`。评测输出存在时默认拒绝覆盖；并发使用同一张卡的本工具进程会被锁阻止。

换未来断点可传 `--checkpoint /absolute/path/checkpoint-N --expected-step N`；E1/E2 会检查 memory 配置匹配，VLM 必须保持 current_frame_only/subtask，预测 horizon=30。当前 v1 协议固定本轮 train40/872 锚点与 batch=4，便于和 1k 严格比较。扩展别的 split 时应另建协议，明确修改相应基准和样本数校验。

## 查看结果

- `E{1,2}/job_status.json`：运行状态和阶段；`openloop/progress.json`：开环进度。
- `E{1,2}/openloop.log`、`success.log`、`*_launcher.log`：日志。
- `E{1,2}/comparison.json`、`REPORT_zh.md`、`action_comparison.{png,svg}`：1k/当前断点比较。
- `E{1,2}/openloop/predictions.npz`：预测、GT、state、有效 mask、episode/frame 等原始数据。
- `E{1,2}/openloop/per_prediction.jsonl`、`per_prediction_summary.csv`：每次预测；`per_action_errors_*.csv.gz`：每个动作的七关节有符号误差和 TCP 距离。
- `E{1,2}/success/results.json`：每任务闭环成功率和有效 episode 数。视频位于 `success/videos/`。
- `E{1,2}/manifest.json`、`checkpoint_metadata/`、`source_snapshot/`：断点元数据、协议、源码与 SHA256。

本次自动启动目录的根 `REPORT_zh.md` / `comparison.json` 会在任一模型完成阶段后刷新。手动单模型运行也会生成该模型的完整报告。

## 指标与公平比较

同一 train40（4 个任务各 10 条）从 GT 轨迹按 stride16 取 872 个独立观测锚点；每次生成 30×8 动作（七个绝对关节角 rad + 夹爪）。主指标统计前 16 步有效动作，排除结尾 padding；MAE 按有效 command/joint 加权。分别测 GT teacher-forced sub-task 与实际 generated sub-task；后者更接近部署条件。两条件使用同一动作随机种子。

TCP command error 为预测/GT 关节角经 Panda URDF 正运动学后的末端位置距离，单位 mm；它不能等同于真实闭环累计轨迹误差。另跑完整 train40 闭环成功率，使用模型生成 sub-task、ODE10、temperature1、执行16、group5、max_steps1300、bf16/FA2、seed0。

1k 基准：`output/fullpretrain_sf_posttrain_20261025_v1/action_error_step1000_20261025_215329/`（开环）及 `eval_step1000_train40_20261025_201803/`（成功率）。报告自动核对 episode/frame、GT/state、有效 mask、stats 和 flow timesteps 一致；还用保存的 1k 原始数组重算旧指标作为数值回归。

开环 generated 的 1k 基准：E1 关节 MAE 1.529°、TCP 26.505mm；E2 1.063°、18.710mm。1k 成功率 E1 1/40=2.5%，E2 3/40=7.5%。本轮只覆盖训练集抽样，不代表官方 test。

评测无需 W&B；训练的 W&B 记录保持原配置。本轮日志、报告、图和数值数组镜像到 Mac `/Users/liam/Projects/research/mem-vla/experiments/fullpretrain_sf_posttrain_20261025_v1/`，权重/优化器和视频不自动下载。
