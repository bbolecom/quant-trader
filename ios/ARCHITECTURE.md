# iOS 架构说明

美股量化 App v3.0 — SwiftUI 壳 + LAN JSON + Streamlit WebView。

## 目录结构

```
Sources/
├── App.swift                 # @main · 注入全局 EnvironmentObject
├── ContentView.swift         # 5 Tab 根导航
├── AppServices.swift         # refreshAll() 统一刷新
├── FeatureRouter.swift       # 功能 → 目标页（唯一路由）
├── CatalogEnrichment.swift   # 策略目录回填（对齐 Python）
├── FeatureHubView.swift      # Tab「功能」+ 分类列表
├── HomeHubView.swift         # Tab「首页」同花顺式 Feed
├── DailyPickView.swift       # Tab「选股」
├── StrategyCatalogView.swift # 全策略目录
├── ModuleDetailView.swift    # 单模块 JSON 详情
├── PickDetailView.swift      # 选股行详情 Sheet
├── ProfileView.swift         # Tab「我的」· 服务器配置
├── AppManifest.swift         # manifest 模型 + ManifestLoader.shared
├── DailyPickModels.swift     # daily_pick 模型 + DailyPickLoader.shared
├── AppSettings.swift         # JSON/Streamlit URL + AppNavigation
├── Config.swift              # 编译期默认 URL
├── JsonDataLoader.swift      # 模块 JSON 通用加载
├── Theme.swift / ThsComponents.swift
└── WebView.swift             # Tab「终端」
```

## 数据流

```
Mac: daily_pick.py + research/*.json (HTTP :8502)
         │
         ▼
┌─────────────────────┐     ┌──────────────────────┐
│ DailyPickLoader     │     │ ManifestLoader       │
│ daily_pick_today    │     │ app_manifest.json    │
└─────────┬───────────┘     └──────────┬───────────┘
          │                            │
          └──────────┬─────────────────┘
                     ▼
              AppServices.refreshAll()
                     │
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
 HomeHubView   DailyPickView   FeatureHubView
                     │
                     ▼
           CatalogEnrichment（回填 modules_summary）
                     │
                     ▼
           StrategyCatalogView → FeatureRouter
```

### 两个 JSON 源

| 文件 | 用途 |
|------|------|
| `app_manifest.json` | 功能元数据：名称、图标、view_type、today_json 路径 |
| `daily_pick_today.json` | 今日选股、picks、modules_summary、strategy_summary |

策略目录展示时：**catalog 行统计** 由 `CatalogEnrichment` 从 `modules_summary` 回填，与 Streamlit 端逻辑一致。

## Tab 与导航

| Tab | 视图 | 职责 |
|-----|------|------|
| 0 首页 | `HomeHubView` | 金刚区、大盘 Banner、热点 Feed |
| 1 功能 | `FeatureHubView` | 按分类浏览全部 manifest 功能 |
| 2 选股 | `DailyPickView` | 高胜率可执行、模块信号、策略目录入口 |
| 3 终端 | `WebView` | Streamlit 全功能 |
| 4 我的 | `ProfileView` | JSON / Streamlit 地址、刷新 |

跨 Tab：`AppNavigation`（`openTerminal()` / `openPicks()` / `openHome()`）

## 功能路由（唯一入口）

所有「点进某个策略」的路径统一走 `FeatureDestinationView`：

```
daily_pick        → DailyPickView(embedded)
terminal_only     → TerminalFeatureView → 跳转 Tab 3
其他              → ModuleDetailView(feature)
```

策略目录行走 `StrategyRowDestinationView`：先用 manifest 查元数据，再合并 catalog 统计。

## 配置

- **运行时**：「我的」Tab 填写 JSON 基址 + Streamlit URL（`AppSettings`）
- **编译期默认**：`Config.swift`（仅兜底）
- JSON 基址自动推导：Streamlit `:8501` → JSON `:8502`

## 离线兜底

1. 远程 JSON（LAN）
2. 内置 `Resources/app_manifest.json`
3. `ManifestLoader.fallbackManifest`（空功能列表 + 分类骨架）

## 与 Python 同步

```bash
python -m quant.app_manifest   # → research/app_manifest.json
cp research/app_manifest.json ios/Resources/app_manifest.json
python daily_pick.py           # → research/daily_pick_today.json
```

## 已移除的遗留

- v2.1「策略」Tab、`StrategyCatalogStaticView`
- `SettingsView`（设置统一到「我的」）
- 策略目录硬编码 json 路径（改由 manifest 提供）
