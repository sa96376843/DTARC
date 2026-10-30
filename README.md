# DTARC：修正实现与可复现汇总结果

当前科学版本：2026-10-31。保留漂移类型感知回放的原定义，修正实现与时间可得性后重选参数、完整重跑并整体更新结果。旧研究快照保存在仓库历史；不要混用旧指标与当前代码。

## 修正内容与证据

- 检测误差使用预测当时保存的概率；标签实际成熟后才可学习。
- 市场上下文改为同日跨股票均值；LOO 仅移除自身贡献；执行对齐面板保留相同主特征。
- 中国五日验证剔除跨入 2026 的 60 个标签；DTARC 回放去重及补齐均保证最终索引唯一。
- 预先固定 10,000 次 moving-block bootstrap，保留原 24 项 Holm 家族和跨 seed 独立日期块设计；修正买入持有初始成本及持有间隔口径。
- 所有 11 组预测文件独立重算，674 行汇总指标一致。另从独立目录重新训练全部 80 个主实验组合，预测、指标和漂移事件逐字节一致。此复现结论只对应新版本及核验环境，不表示恢复了旧冻结神经网络输出。

## 结果边界

中国 DTARC MCC/AUC 为 0.0634/0.5404，泰国为 −0.0476/0.4746。三个 AUC 比较通过 Holm：DTARC 在两市场优于周期回放，在中国低于静态网络。与均匀 ADWIN 回放的主比较均不显著。主 ADWIN 配置下 gradual 分支结构上不可达，不能据此验证两分支机制；10 bps 成本下主多空结果均为负。

## 文件与执行

`src/` 为修正代码，`results/` 为聚合结果，`data/dataset_summary.json` 为面板元数据。`VERIFICATION.json` 记录核验，`DATA_SNAPSHOT_HASHES.json` 提供本地快照哈希，`SOURCE_MANIFEST.json` 校验公开文件。

无需科学依赖的发布完整性检查：

```bash
python verify_public.py
```

公开子集不含原始行情、逐日面板、逐行预测、经济曲线或初始化权重。需按 `DATA_SOURCES.md` 合法取得兼容数据，先复制到独立目录，再依次执行：

```bash
python -m pip install -r requirements.txt
python src/run_pipeline.py --stage data
python src/run_pipeline.py --stage experiments
python src/run_pipeline.py --stage artifacts
python src/run_pipeline.py --stage validate
```

运行会覆盖独立工作副本的对应输出。`experiments` 使用 10,000 次 bootstrap；`artifacts` 仅生成 15 张实证图和数值表，已禁止旧入口覆盖正式稿、图 1/图 2 或作者资料。正式稿件属于另一个带版本的作者源包，以 `main_plos.tex` 为唯一正文源；不在此公开子集中。完整依赖清单见 `requirements-lock.txt`，主要版本与环境见 `ENVIRONMENT.txt`。

上游历史会变化；本公开子集不能单独保证离线、逐字节重建本地快照。直接运行完整数据校验而未取得文件，会报明确的缺失错误。数据共享与再分发许可仍需作者核实；不推定公开接口等于可再分发。未新增开源许可证。
