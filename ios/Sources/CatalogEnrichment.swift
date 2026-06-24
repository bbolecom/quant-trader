import Foundation

/// 与 Python `enrich_catalog_from_daily_pick` 对齐：用 modules_summary 回填策略目录统计。
enum CatalogEnrichment {
    private static let moduleAliases: [String: String] = [
        "暴涨80%": "gain15",
        "暴涨80%·回避": "gain15",
        "暴涨80%·观察": "gain15",
        "5×舰队·CSP": "fleet_csp",
        "资金流向": "capital_flow",
        "规律·Ultra80": "meme_pattern",
        "收入·卖Call": "bear_call",
        "弱市·卖Call": "bear_call",
        "三腿策略": "pattern_daily",
        "pattern_daily": "pattern_daily",
        "flow_strategy": "flow_strategy",
        "VRP波动率": "vrp",
        "VRP波动率·CSP": "vrp",
        "日历价差": "calendar",
        "SNDK铁鹰": "sndk_iron",
        "strategy_rank": "strategy_rank",
        "screen_daily": "screen_daily",
        "自选股扫描": "scan_daily",
        "universal_playbook": "universal_playbook",
        "高频·动量": "trajectory_highwin",
        "轨迹·高置信": "trajectory_highwin",
    ]

    static func enrichedCatalog(from document: DailyPickDocument) -> [StrategyCatalogRow] {
        guard let rows = document.strategySummary?.catalog, !rows.isEmpty else { return [] }
        let mods = document.modulesSummary ?? [:]
        let summary = document.summary
        let pickDate = document.pickDate ?? "—"
        var byID = Dictionary(uniqueKeysWithValues: rows.map { ($0.strategyID, $0) })

        for (modKey, stats) in mods {
            guard let sid = strategyID(for: modKey, catalog: rows) else { continue }
            guard var row = byID[sid] else { continue }
            row = row.with(
                actionable: row.actionable + (stats.actionable ?? 0),
                watching: row.watching + (stats.watching ?? 0),
                total: row.total + (stats.total ?? 0),
                hasData: true,
                dataDate: pickDate
            )
            byID[sid] = row
        }

        if var hub = byID["daily_pick"], let summary {
            hub = hub.with(
                actionable: summary.actionable ?? hub.actionable,
                watching: summary.watching ?? hub.watching,
                total: summary.total ?? hub.total,
                hasData: true,
                dataDate: pickDate
            )
            byID["daily_pick"] = hub
        }

        for run in document.moduleRuns ?? [] {
            guard var row = byID[run.moduleID] else { continue }
            row = row.with(hasData: true, dataDate: pickDate)
            if let count = run.rowCount {
                row = row.with(total: max(row.total, count))
            }
            byID[run.moduleID] = row
        }

        return rows.map { byID[$0.strategyID] ?? $0 }
    }

    private static func strategyID(for modKey: String, catalog: [StrategyCatalogRow]) -> String? {
        if let sid = moduleAliases[modKey] { return sid }
        for row in catalog {
            let label = row.moduleLabel
            if !label.isEmpty, label != "—", modKey.contains(label) || modKey.hasPrefix(label) {
                return row.strategyID
            }
        }
        return nil
    }
}

extension StrategyCatalogRow {
    func with(
        actionable: Int? = nil,
        watching: Int? = nil,
        total: Int? = nil,
        hasData: Bool? = nil,
        dataDate: String? = nil
    ) -> StrategyCatalogRow {
        StrategyCatalogRow(
            strategyID: strategyID,
            name: name,
            category: category,
            integrated: integrated,
            moduleLabel: moduleLabel,
            hasData: hasData ?? self.hasData,
            actionable: actionable ?? self.actionable,
            watching: watching ?? self.watching,
            total: total ?? self.total,
            dataDate: dataDate ?? self.dataDate,
            detail: detail
        )
    }

    /// 合并 manifest 元数据与 catalog 行上的实时统计。
    func resolvedFeature(manifest: AppManifest?) -> ManifestFeature {
        if let base = manifest?.features.first(where: { $0.id == strategyID }) {
            return base.merging(catalog: self)
        }
        return fallbackFeature()
    }

    private func fallbackFeature() -> ManifestFeature {
        ManifestFeature(
            id: strategyID,
            name: name,
            category: category,
            thsCategory: thsCategoryId(for: category),
            icon: "square.grid.2x2",
            script: nil,
            config: nil,
            todayJson: nil,
            todayCsv: nil,
            historyCsv: nil,
            description: detail,
            integratedInDailyPick: integrated,
            dailyPickModule: moduleLabel == "—" ? nil : moduleLabel,
            launcher: nil,
            viewType: "json_generic",
            terminalTab: nil,
            actionable: actionable,
            watching: watching,
            total: total,
            hasData: hasData,
            dataDate: dataDateLabel == "—" ? nil : dataDateLabel
        )
    }

    private func thsCategoryId(for cat: String) -> String {
        switch cat {
        case "聚合": return "hub"
        case "动量": return "momentum"
        case "量价": return "flow"
        case "规律": return "pattern"
        case "期权收入", "期权": return "options"
        case "综合": return "composite"
        case "筛选", "监控": return "screen"
        default: return "composite"
        }
    }
}

extension ManifestFeature {
    func merging(catalog row: StrategyCatalogRow) -> ManifestFeature {
        ManifestFeature(
            id: id,
            name: row.name.isEmpty ? name : row.name,
            category: row.category.isEmpty ? category : row.category,
            thsCategory: thsCategory,
            icon: icon,
            script: script,
            config: config,
            todayJson: todayJson,
            todayCsv: todayCsv,
            historyCsv: historyCsv,
            description: row.detail.isEmpty ? description : row.detail,
            integratedInDailyPick: row.integrated,
            dailyPickModule: row.moduleLabel == "—" ? dailyPickModule : row.moduleLabel,
            launcher: launcher,
            viewType: viewType,
            terminalTab: terminalTab,
            actionable: row.actionable,
            watching: row.watching,
            total: row.total,
            hasData: row.hasData,
            dataDate: row.dataDateLabel == "—" ? dataDate : row.dataDateLabel
        )
    }
}
