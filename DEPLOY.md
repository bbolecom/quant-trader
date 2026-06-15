# 部署指南

把本项目从「本地开发」推到「网页随处访问 + iPhone App 加载」，按下面顺序做即可。

---

## 第 0 步：确认本地正常

```bash
cd /Users/Admin/Desktop/666
./run.sh                    # 浏览器打开 http://localhost:8501
pytest                      # 74 项测试应全部通过
```

---

## 第 1 步：推到 GitHub

### 方式 A：网页创建仓库（无需 gh 命令）

1. 打开 [github.com/new](https://github.com/new)，新建仓库（例如 `us-quant-trader`），**不要**勾选「Add README」（本地已有）。
2. 在终端执行（把 URL 换成你的）：

```bash
cd /Users/Admin/Desktop/666
git remote add origin https://github.com/你的用户名/us-quant-trader.git
git push -u origin main
```

### 方式 B：安装 GitHub CLI 后一键创建

```bash
brew install gh
gh auth login
cd /Users/Admin/Desktop/666
./scripts/setup_github.sh us-quant-trader
```

---

## 第 2 步：部署 Streamlit Community Cloud（免费 HTTPS）

1. 打开 [share.streamlit.io](https://share.streamlit.io) → 用 GitHub 登录。
2. **New app** → 选择你的仓库、分支 `main`、主文件 **`app.py`**。
3. **Advanced settings**（可选）：
   - Python version: **3.11**
4. 点击 **Deploy**，等待 2–5 分钟。
5. 部署成功后得到地址，例如：`https://us-quant-trader.streamlit.app`

> 首次打开若较慢，是免费版休眠唤醒，属正常现象。  
> 依赖按 `requirements.txt` 自动安装；行情走 Yahoo Finance，云端可正常联网。

### 常见问题

| 现象 | 处理 |
|---|---|
| 部署失败 / ModuleNotFoundError | 确认 `requirements.txt` 已提交到 GitHub |
| 页面空白或 503 | 查看 Streamlit Cloud 日志（Manage app → Logs） |
| 拉不到行情 | 检查 Yahoo Finance 是否被墙；Cloud 一般可用 |
| 想换主题 | 改 `.streamlit/config.toml` 后 push，Cloud 会自动重建 |

---

## 第 3 步：装进 iPhone / iPad

1. **先把第 2 步的 Streamlit 地址记下来**。
2. 编辑 `ios/Sources/Config.swift`：

```swift
static let serverURLString = "https://你的应用名.streamlit.app"
```

3. 生成 Xcode 工程并打开：

```bash
brew install xcodegen   # 只需一次
cd ios && xcodegen generate && open QuantTrader.xcodeproj
```

4. Xcode：**Signing & Capabilities** → 选 Team、改唯一 Bundle ID → 连设备 → **Run**。
5. 设备上：**设置 → 通用 → VPN 与设备管理** → 信任开发者。

详细说明见 [`ios/README.md`](ios/README.md)。

---

## 第 4 步：定时信号扫描（可选）

在 **Mac 本机**运行（Cloud 上不适合跑 cron）：

```bash
source .venv/bin/activate
python scan_daily.py --dry-run    # 试跑
python scan_daily.py              # 正式扫描 + 桌面通知
```

定时任务见根目录 `com.quant.scan.plist` 与 `README.md`「定时自动扫描」一节。

---

## 环境变量与密钥

- 本 Web 应用**不需要** API Key（行情来自 Yahoo Finance）。
- 邮件通知：在 `scan_config.json` 里配置 SMTP，密码放环境变量 `SCAN_EMAIL_PASSWORD`（**不要提交到 Git**）。
- 若将来需要 secrets，在 Streamlit Cloud：**App settings → Secrets**，格式同 `.streamlit/secrets.toml`（可参考 `.streamlit/secrets.toml.example`，若存在）。

---

## 推送后自动测试

仓库已配置 GitHub Actions（`.github/workflows/test.yml`）：每次 push / PR 到 `main` 会自动跑 `pytest`。在 GitHub 仓库 **Actions** 页可查看结果。
