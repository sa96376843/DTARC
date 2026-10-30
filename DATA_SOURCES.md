# 数据来源与快照口径

## 中国主面板

- 当前主结果使用新浪财经同一来源、同一前复权口径的完整历史序列。
- 原始快照：`data/cn_raw_sina_qianfuquan.csv`。
- 处理后主面板：`data/cn_processed.csv`，与 `cn_processed_sina_qianfuquan.csv` 内容一致。
- 后复权稳健性快照：`data/cn_raw_sina_houfuquan.csv` 和 `data/cn_processed_sina_houfuquan.csv`。
- 下载代码：`src/download_sina_adjustment_audit.py`。
- 代码中使用的公开页面与接口包括：
  - `https://finance.sina.com.cn/`
  - `https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData`
  - `https://finance.sina.com.cn/realstock/company/{symbol}/qianfuquan.js`
  - `https://finance.sina.com.cn/realstock/company/{symbol}/houfuquan.js`

## 泰国面板

- 来源：SiamChart 公开历史行情页面与接口。
- 原始合并快照：`data/th_raw_siamchart.csv`；分股票快照为 `data/th_*.csv`。
- 处理后面板：`data/th_processed.csv`。
- 下载代码：`src/download_th_parallel.py`；较早的顺序下载实现保留在 `src/download_data.py`。
- 公开端点形式：`https://siamchart.com/query/history?symbol={symbol}&resolution=D&from={start}&to={end}`。
- 页面形式：`https://siamchart.com/stock-chart/{symbol}/`。

## 被否决的跨源拼接面板与失败审计

- `data/cn_raw_baostock.csv` 是早期中国历史段及后续拼接流程使用的审计快照，不是当前主面板的数据口径。
- `data/cn_processed_stitched_audit.csv` 及 `results/boundary_*_stitched.csv` 用于展示跨来源复权尺度不一致造成的机械跳变。
- `src/download_data.py` 和 `src/download_cn_2026.py` 记录该历史流程；重新运行它们可能覆盖审计快照，因此复现当前论文时无需运行。
- BaoStock 仅用于可选的历史审计下载，额外依赖见 `requirements-download.txt`。

## 时间范围与划分

- 原始序列从 2023-11-13 起用于特征预热。
- 共同建模期为 2023-12-11 至 2026-09-30。
- 2023–2024：初始训练；2025：中国验证；2026-01-01 至 2026-09-30：测试。
- 每市场 12 只股票，共 24 只；这是方法审计样本，不是市场代表性抽样或资产定价因子研究样本。
- 精确股票代码、行数、划分和五日上涨比例见 `data/dataset_summary.json`。

## 使用注意

这些文件是为学术复现保存的公开行情快照。公开接口可能更新、限流或失效；重新下载得到的未来快照不一定与本文相同。使用者应遵守数据来源网站当时适用的条款，不得把接口失败解释为空结果，也不应绕过认证或访问限制。本文与代码不构成投资建议。
