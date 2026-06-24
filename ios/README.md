# 把量化平台装进 iPhone / iPad（原生 App）

整体思路：**Streamlit 应用 = 需要一个服务器在跑**。所以分两步：

1. **部署后端**：把这个 Python 应用跑在一个能用网址访问的地方（云端或你自己的电脑）。
2. **装壳 App**：用这里的 SwiftUI + WKWebView 工程，打包成 iOS App 装到你的设备，App 打开后加载第 1 步的网址。

---

## 第 1 步：部署后端，拿到一个网址

### 方案 A：Streamlit Community Cloud（推荐，免费、HTTPS、随处可访问）

1. 把整个项目（`666` 文件夹）推到一个 GitHub 仓库（公开或私有都行）。
2. 打开 <https://share.streamlit.io> → 用 GitHub 登录 → **New app**。
3. 选择你的仓库、分支，主文件填 `app.py` → **Deploy**。
4. 部署完成后会得到一个网址，形如 `https://你的应用名.streamlit.app`。
5. 记下这个网址，第 3 步要用。

> 依赖会自动按 `requirements.txt` 安装；行情来自 Yahoo Finance，云端可正常联网获取。

### 方案 B：自己电脑/局域网自托管（免费，但需电脑开着、与手机同 Wi-Fi）

```bash
cd /Users/Admin/Desktop/666
source .venv/bin/activate
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

查到你 Mac 的局域网 IP（系统设置 → 网络，或 `ipconfig getifaddr en0`），地址即
`http://你的IP:8501`（例如 `http://192.168.1.20:8501`）。手机需与电脑在同一 Wi-Fi。

> 自托管走的是 http，本工程已在 `Info.plist` 里保留了 ATS 例外，可直接加载。

---

## 第 2 步：生成 Xcode 工程

需要先装好 **Xcode**（App Store 免费下载）。

为避免手写易坏的工程文件，这里用 **XcodeGen** 一键生成：

```bash
# 安装 xcodegen（只需一次）
brew install xcodegen

# 在 ios 目录生成工程
cd /Users/Admin/Desktop/666/ios
xcodegen generate
```

会生成 `QuantTrader.xcodeproj`（已自动包含内置的 App 图标）。

> 没装 Homebrew？先执行：`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

---

## 第 3 步：填入你的服务地址

**推荐**：安装后在 App **「我的」** Tab 填写 JSON 基址与 Streamlit 地址，无需改代码。

编译期默认（可选）在 `ios/Sources/Config.swift`：

```swift
static let serverURLString = "https://你的应用名.streamlit.app"
// 或自托管： "http://192.168.1.20:8501"
```

---

## 第 4 步：在 Xcode 里运行到设备

1. 打开工程：`open QuantTrader.xcodeproj`
2. 左侧选中项目 → TARGETS → **QuantTrader** → **Signing & Capabilities**：
   - 勾选 **Automatically manage signing**
   - **Team** 选择你的 Apple ID（个人免费账号即可，见下方说明）
   - **Bundle Identifier** 改成全局唯一的，例如 `com.你的名字.quanttrader`
3. 用数据线连接 iPhone/iPad，顶部设备选择器选中你的设备。
4. 点 **▶︎ Run**。
5. 首次安装后，在设备上：**设置 → 通用 → VPN 与设备管理 → 信任你的开发者证书**，再打开 App。

完成后桌面就会出现「美股量化」App 图标，点开即用。

---

## v3.0 同花顺式全功能 App

App 升级为 **五 Tab** 结构，系统内 **30+ 功能模块** 全部原生入口：

| Tab | 功能 |
|-----|------|
| **首页** | 大盘 Regime · 四宫格指标 · 金刚区快捷入口 · 8 大分类 · 模块运行状态 |
| **功能** | 全系统功能清单（聚合/动量/量价/规律/期权/综合/筛选/实验室） |
| **选股** | 高胜率≥80% · 观察池 · 模块信号 · 策略目录 |
| **终端** | Streamlit 全平台（回测/寻优/期权/模拟盘等 18 个 Tab） |
| **我的** | **可编辑** JSON 基址 + Streamlit 地址 · 刷新 · 连接说明 |

### 原生支持的模块 JSON

每个功能点可点击进入 **ModuleDetailView**，自动拉取对应 `today.json`：

- 暴涨80%规则 · 暴涨扫描 A/B/C · SPCE投机池
- 资金流向 · Meme规律 · 三腿策略 · Universal舰队
- 以及 strategy_catalog 登记的全部模块

功能清单文件：`research/app_manifest.json`（内置在 App Bundle，离线可浏览）

### 连接 Mac（推荐局域网）

```bash
cd /Users/Admin/Desktop/666
./run.sh   # 默认启动 8501 Streamlit + 8502 JSON 静态服务
python daily_pick.py   # 刷新全部 today.json
python -m quant.app_manifest   # 单独导出功能清单
```

在 App **「我的」** Tab 填写：

- **JSON 基址**：`http://你的MacIP:8502/`（注意末尾斜杠）
- **Streamlit**：`http://你的MacIP:8501`

无需再改代码重新编译（除非换默认云端地址）。

**架构说明**：见 [`ARCHITECTURE.md`](ARCHITECTURE.md)（目录结构、数据流、路由规则）。

---

## 关于签名 / 账号

- **免费 Apple ID**：可以装到自己的真机，但证书 **7 天过期**，过期后需在 Xcode 重新 Run 一次；同时安装的自签 App 数量有限。
- **付费开发者账号（$99/年）**：证书有效期 1 年，并可用 **TestFlight** 远程安装、分发给他人，体验更顺。

自用、且能接受每周重签的话，免费账号就够了；想省心或分享给别人，再买付费账号。

---

## 常见问题

- **App 打开是「无法连接到服务」**：检查「我的」里 Streamlit 地址、后端是否在跑；自托管时确认手机与电脑同一 Wi-Fi。
- **选股 Tab 无数据**：在「我的」填 JSON 基址 `http://MacIP:8502/`，Mac 运行 `python daily_pick.py` 后点「刷新全部数据」。
- **Streamlit Cloud 应用休眠**：免费版闲置会休眠，首次打开需等十几秒唤醒，属正常现象。
- **App 图标**：已内置一套量化风格图标（`Sources/Assets.xcassets/AppIcon.appiconset/icon-1024.png`），生成工程后自动生效。想换图标，替换这张 1024×1024 PNG 即可。
- **横竖屏**：已支持，iPad 体验更佳。

> ⚠️ 本工具仅供个人研究，所有回测/评分/概率均为历史统计，不构成投资建议。
