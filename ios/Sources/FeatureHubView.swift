import SwiftUI

/// Tab「功能」：按 manifest 分类浏览全部策略。
struct FeatureHubView: View {
    @EnvironmentObject private var manifestLoader: ManifestLoader

    var body: some View {
        NavigationStack {
            Group {
                if manifestLoader.isLoading && manifestLoader.manifest == nil {
                    ProgressView("加载功能清单…")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List(manifestLoader.manifest?.categories ?? []) { cat in
                        let items = manifestLoader.features(in: cat.id)
                        if !items.isEmpty {
                            NavigationLink {
                                FeatureCategoryView(category: cat, features: items)
                            } label: {
                                ThsCategoryRow(
                                    category: cat,
                                    count: items.count,
                                    actionable: items.reduce(0) { $0 + ($1.actionable ?? 0) }
                                )
                            }
                            .listRowBackground(ThsTheme.card)
                        }
                    }
                    .scrollContentBackground(.hidden)
                }
            }
            .background(ThsTheme.background)
            .navigationTitle("功能")
            .navigationBarTitleDisplayMode(.large)
            .task {
                if manifestLoader.manifest == nil {
                    await manifestLoader.reload()
                }
            }
            .refreshable { await manifestLoader.reload() }
        }
        .preferredColorScheme(.dark)
    }
}

/// 分类下功能列表。
struct FeatureCategoryView: View {
    let category: ManifestCategory
    let features: [ManifestFeature]
    @State private var query = ""

    private var filtered: [ManifestFeature] {
        let q = query.trimmingCharacters(in: .whitespaces)
        guard !q.isEmpty else { return features }
        return features.filter {
            $0.name.localizedCaseInsensitiveContains(q)
                || $0.detail.localizedCaseInsensitiveContains(q)
        }
    }

    var body: some View {
        List(filtered) { feat in
            NavigationLink {
                FeatureDestinationView(feature: feat)
            } label: {
                ManifestFeatureRow(feature: feat)
            }
            .listRowInsets(EdgeInsets(top: 6, leading: 16, bottom: 6, trailing: 16))
            .listRowSeparator(.hidden)
            .listRowBackground(Color.clear)
        }
        .listStyle(.plain)
        .scrollContentBackground(.hidden)
        .background(ThsTheme.background)
        .navigationTitle(category.name)
        .searchable(text: $query, prompt: "搜索\(category.name)")
        .preferredColorScheme(.dark)
    }
}

struct ThsCategoryRow: View {
    let category: ManifestCategory
    let count: Int
    let actionable: Int

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: category.icon)
                .font(.title3)
                .foregroundStyle(ThsTheme.accent)
                .frame(width: 36)
            VStack(alignment: .leading, spacing: 4) {
                Text(category.name)
                    .font(.headline)
                    .foregroundStyle(ThsTheme.textPrimary)
                Text("\(count) 项功能")
                    .font(.caption)
                    .foregroundStyle(ThsTheme.textSecondary)
            }
            Spacer()
            if actionable > 0 {
                Text("\(actionable) 信号")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(ThsTheme.up)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(ThsTheme.up.opacity(0.15), in: Capsule())
            }
            Image(systemName: "chevron.right")
                .font(.caption.weight(.bold))
                .foregroundStyle(ThsTheme.textTertiary)
        }
        .padding(14)
        .thsCard()
    }
}

struct ManifestFeatureRow: View {
    let feature: ManifestFeature

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: feature.icon)
                    .foregroundStyle(ThsTheme.accent)
                Text(feature.name)
                    .font(.headline)
                Spacer()
                if feature.integrated {
                    Text("已接入")
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(ThsTheme.up)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(ThsTheme.up.opacity(0.12), in: Capsule())
                }
            }
            if !feature.detail.isEmpty {
                Text(feature.detail)
                    .font(.footnote)
                    .foregroundStyle(ThsTheme.textSecondary)
                    .lineLimit(2)
            }
            HStack(spacing: 12) {
                label("可开", feature.actionable ?? 0, ThsTheme.up)
                label("观望", feature.watching ?? 0, .orange)
                if feature.hasJsonFeed {
                    Text(feature.hasData == true ? "有数据" : "无今日JSON")
                        .font(.caption2)
                        .foregroundStyle(feature.hasData == true ? ThsTheme.accent : ThsTheme.textTertiary)
                } else if feature.isTerminalOnly {
                    Text("量化终端")
                        .font(.caption2)
                        .foregroundStyle(Color.cyan)
                }
            }
        }
        .padding(14)
        .thsCard(border: (feature.actionable ?? 0) > 0 ? ThsTheme.up.opacity(0.25) : ThsTheme.border)
    }

    private func label(_ title: String, _ n: Int, _ color: Color) -> some View {
        HStack(spacing: 3) {
            Text("\(n)").font(.caption.weight(.bold))
            Text(title).font(.caption2)
        }
        .foregroundStyle(n > 0 ? color : ThsTheme.textTertiary)
    }
}

#Preview {
    FeatureHubView()
        .environmentObject(ManifestLoader.shared)
        .environmentObject(AppNavigation())
}
