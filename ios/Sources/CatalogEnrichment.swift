import Foundation

/// 与 Python `enrich_catalog_from_daily_pick` 对齐：用 modules_summary 回填策略目录统计。
enum CatalogEnrichment {
    private static let moduleAliases: [String: String] = [
        "暴涨80%": "gain15",
        "暴涨80%·回避": "gain15",
        "暴涨80%·观察": "gain15",
        "暴涨80%·新暴涨": "gain15",
        "Extreme20": "extreme20",
        "Extreme20·L1": "extreme20",
        "Extreme20·S1": "extreme20",
        "Extreme20·L2": "extreme20",
        "Extreme20·S2": "extreme20",
        "5×舰队·CSP": "fleet_csp",
        "资金流向": "capital_flow",
        "规律·Ultra80": "meme_long",
        "规律·纯多头": "meme_long",
        "收入·卖Call": "bear_call",
        "弱市·卖Call": "bear_call",
        "flow_strategy": "flow_strategy",
        "资金流向组合": "flow_strategy",
        "VRP波动率": "vrp",
        "VRP波动率·CSP": "vrp",
        "SNDK铁鹰": "sndk_iron",
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

        let legacyAliases: [String: String] = ["meme_pattern": "meme_long"]
        for (oldID, newID) in legacyAliases where byID[newID] == nil {
            if let legacy = byID[oldID] {
                byID[newID] = StrategyCatalogRow(
                    strategyID: newID,
                    name: legacy.name,
                    category: legacy.category,
                    integrated: legacy.integrated,
                    moduleLabel: legacy.moduleLabel,
                    hasData: legacy.hasData,
                    actionable: legacy.actionable,
                    watching: legacy.watching,
                    total: legacy.total,
                    dataDate: legacy.dataDate,
                    detail: legacy.detail
                )
            }
        }

        return coreOrder.compactMap { byID[$0] }
    }

    private static let coreOrder: [String] = [
        "daily_pick", "capital_flow", "flow_strategy", "meme_long",
        "gain15", "extreme20", "bear_call", "fleet_csp", "sndk_iron", "vrp",
    ]

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
            dataDate: dataDateLabel == "—" ? nil : dataDateLabel,
            trades: nil,
            winRate: nil,
            annReturn: nil,
            maxDd: nil,
            sharpe: nil,
            auditRank: nil,
            auditScore: nil,
            auditTier: nil,
            auditVerdict: nil,
            auditAction: nil
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
            dataDate: row.dataDateLabel == "—" ? dataDate : row.dataDateLabel,
            trades: trades,
            winRate: winRate,
            annReturn: annReturn,
            maxDd: maxDd,
            sharpe: sharpe,
            auditRank: auditRank,
            auditScore: auditScore,
            auditTier: auditTier,
            auditVerdict: auditVerdict,
            auditAction: auditAction
        )
    }
}
