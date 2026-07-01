# 2026 世界杯小红书金球竞猜投资分析器

这是一个基于 Streamlit 的本地数据应用，可以读取仓库内的 Excel 数据文件，输出单场推荐、今日组合建议、风险调整后胜率、风险调整后 EV 和 risk-adjusted score。

项目已经整理为适合上传到 GitHub 并部署到 Streamlit Community Cloud 的结构，不依赖本机绝对路径，也不联网、不爬取小红书、不重新训练模型。

## 项目结构

根目录保留这些核心文件：

- `app.py`
- `data_loader.py`
- `model.py`
- `portfolio.py`
- `requirements.txt`
- `README.md`

数据文件统一放在：

- `data/`

测试文件放在：

- `tests/`

## 数据文件

请将以下 Excel 文件放入 `data/` 目录：

- `2026世界杯投资分析_未来比赛特征表_complete (1).xlsx`
- `2026世界杯投资分析_近期状态汇总表_current_complete.xlsx`
- `2026世界杯小红书竞猜_场外因素与资本力量统计表.xlsx`
- `2026世界杯投资分析_第一阶段与静态实力表.xlsx`

程序通过相对路径 `./data/` 读取数据。

## 本地运行

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 启动应用

```bash
streamlit run app.py
```

如果本机 `streamlit` 没有加入 PATH，也可以用：

```bash
py -3 -m streamlit run app.py
```

## 上传到 GitHub

建议把以下内容上传到 GitHub 仓库：

- `app.py`
- `data_loader.py`
- `model.py`
- `portfolio.py`
- `requirements.txt`
- `README.md`
- `data/`
- `tests/`
- `.gitignore`

不建议上传以下本地辅助内容：

- `core/`
- `.pytest_cache/`
- `__pycache__/`
- `streamlit.log`
- `streamlit.err.log`
- `streamlit_v2.log`
- `streamlit_v2.err.log`
- `streamlit_v3.log`
- `streamlit_v3.err.log`
- 根目录中重复的 xlsx / docx 草稿文件

## 部署到 Streamlit Community Cloud

1. 登录 Streamlit Community Cloud
2. 点击 `New app` 或 `Deploy an app`
3. 选择你的 GitHub 仓库
4. 选择分支
5. 在入口文件位置填写：

```text
app.py
```

6. 点击 `Deploy`
7. 部署成功后，复制生成的 `*.streamlit.app` 链接分享给其他用户

## 依赖

`requirements.txt` 只保留必要依赖：

- `streamlit`
- `pandas`
- `numpy`
- `openpyxl`
