# E1 / E2：配置驱动的 val40 + test200 闭环评测

在服务器 `/mnt/cpfs/heping/SimpleMemVLA` 中运行。入口只需配置文件和 GPU 编号，不需要手工指定 E1/E2、memory 开关或 step。两份配置除 checkpoint 路径外完全一致，共用同一评测协议。

## 配置与数据集

- E1：`/mnt/cpfs/heping/SimpleMemVLA/configs/eval/posttrain_valtest_e1.yaml`
- E2：`/mnt/cpfs/heping/SimpleMemVLA/configs/eval/posttrain_valtest_e2.yaml`
- 公共协议：`/mnt/cpfs/heping/SimpleMemVLA/configs/eval/posttrain_valtest_common.yaml`
- 内部 val40 清单：`/mnt/cpfs/heping/SimpleMemVLA/configs/eval/robomme_heldout_val40_episode_ids.json`

模型配置仅有两行，默认给当前已保存的 16250 步断点：

```yaml
_base_: posttrain_valtest_common.yaml
checkpoint: /mnt/cpfs/heping/SimpleMemVLA-data/output/fullpretrain_sf_50k_fast_uniform_20261026_v2/e1_5w_iter/checkpoint-16250
```

E2 只把路径中的模型目录改为 `e2_5w_iter`。比较另一个阶段时只改 `checkpoint-16250` 为实际存在的断点，例如 `checkpoint-5000`；无需另改 expected_step、实验名或输出目录。也可复制一份配置保存各阶段。

每个模型依次评测：

| 名称 | 环境 split | 数量 | 来源 |
|---|---|---:|---|
| val40 | train | 4任务×10=40 | `robomme_small_val.txt` 中从官方 train 留出的40条，与训练360条不交叉 |
| test200 | test | 4任务×50=200 | PickXtimes、VideoUnmaskSwap、VideoRepick、PatternLock 的全部官方 test episode 0–49 |

这里的 val40 **不是官方 val，也不是旧 test-on-train40**。旧 `robomme_train40_episode_ids.json` 的40条属于训练360，不能用作此次内部验证。新入口在运行前逐条核对内部 val 清单和 train/val 分离，并核对官方冻结元数据的 episode 范围。

## 先检查配置，不运行模型

```bash
cd /mnt/cpfs/heping/SimpleMemVLA
bash scripts/eval_robomme_valtest.sh --config configs/eval/posttrain_valtest_e1.yaml --gpu 6 --dry-run
bash scripts/eval_robomme_valtest.sh --config configs/eval/posttrain_valtest_e2.yaml --gpu 7 --dry-run
```

`--dry-run` 只读配置、checkpoint元数据和清单，打印实际checkpoint、step、memory状态、40/200条任务清单和计划输出目录；不加载模型、不创建评测输出、不运行仿真。它会检查断点是否完整（checkpoint-N需有匹配的trainer_state）。实际运行会生成新的唯一目录，因此其时间戳与dry-run预览可能不同。

## 正式启动

在各自终端直接运行：

```bash
cd /mnt/cpfs/heping/SimpleMemVLA
bash scripts/eval_robomme_valtest.sh --config configs/eval/posttrain_valtest_e1.yaml --gpu 6
```

```bash
cd /mnt/cpfs/heping/SimpleMemVLA
bash scripts/eval_robomme_valtest.sh --config configs/eval/posttrain_valtest_e2.yaml --gpu 7
```

每条命令在一张GPU上顺序完成val40与test200。GPU编号是物理卡号，E1/E2可并行；入口不会启动/停止训练。需要退出SSH后继续时，在服务器同一shell执行：

```bash
cd /mnt/cpfs/heping/SimpleMemVLA
LOGS=/mnt/cpfs/heping/SimpleMemVLA-data/output/robomme_valtest_launcher_logs
mkdir -p "$LOGS"
STAMP=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
nohup bash scripts/eval_robomme_valtest.sh --config configs/eval/posttrain_valtest_e1.yaml --gpu 6 > "$LOGS/e1_$STAMP.log" 2>&1 &
nohup bash scripts/eval_robomme_valtest.sh --config configs/eval/posttrain_valtest_e2.yaml --gpu 7 > "$LOGS/e2_$STAMP.log" 2>&1 &
```

每次运行的输出目录自动生成在 `/mnt/cpfs/heping/SimpleMemVLA-data/output/robomme_valtest/` 下，名称含模型目录、checkpoint目录、时间戳和随机后缀，不覆盖旧结果。启动时的JSON和外层launcher log会打印准确 `output` 路径。

## 结果与恢复

在实际输出目录中查看：

- `summary.json` / `REPORT_zh.md`：两个数据集的成功数、总数和成功率。
- `val40/results.json`、`test200/results.json`：各任务原始成功率与有效episode数。
- `val40/launcher.log`、`test200/launcher.log`：实际模型/仿真日志。
- `job_status.json`：总状态和当前阶段；各阶段 `stage_status.json`：状态与严格校验后的结果。
- `resolved_config.yaml`、`manifest.json`、`source_snapshot/`、各阶段 `episode_ids.json` / `protocol.yaml`：配置、断点身份、代码和清单证据。

只跑一个数据集，加 `--splits val40` 或 `--splits test200`。失败后继续同一轮：

```bash
bash scripts/eval_robomme_valtest.sh --config configs/eval/posttrain_valtest_e1.yaml --gpu 6 --output /实际的旧评测输出目录 --resume
```

保持原配置与原 `--splits` 选择；已完成且校验通过的阶段会复用。checkpoint、配置、代码或episode清单变化后需新建评测，不允许把它们混入旧输出。默认拒绝覆盖非空输出目录，同一入口与旧posttrain入口共用GPU锁。

## 推理口径

保持单帧 VLM、模型生成的 sub-task 条件（闭环不用GT）、ODE10、temperature1、执行前16步、group5、max_steps1300、BF16/FA2、非pipelined。模型加载后采样RNG按现有评测器固定seed0；环境seed/difficulty由官方对应split的冻结metadata恢复。E2从checkpoint自动识别memory，仿真中在线编码已观测历史，只接DiT；不用提供离线训练feature目录。

NA、缺任务、数量不符、OOM标记、无效成功率都会被外层入口判为失败，避免将排除样本后的数字当作完整40/200条结果。该入口复用现有 `robomme_sim.eval_success`，没有改变模型或训练实现。

## 在Mac同步日志

启动服务器评测后，在Mac的终端运行：

```bash
/Users/liam/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 /Users/liam/Projects/research/mem-vla/experiments/valtest_eval_setup_20261027_v1/sync_eval_logs.py
```

每60秒增量同步到 `/Users/liam/Projects/research/mem-vla/experiments/robomme_valtest`；包含配置、指标、源码和日志，跳过视频及模型二进制。评测结束且完成最后一次同步后按Ctrl-C停止；也可加 `--once` 只同步一次。同步状态见目标目录的 `last_sync.json`。

本次仅准备入口/配置，并进行CPU dry-run与调度校验；没有启动真实模型评测，没有新成功率结果。准备材料会同步到 `/Users/liam/Projects/research/mem-vla/experiments/valtest_eval_setup_20261027_v1`，后续结果继续汇总到 `EXP_LOG.md`。
