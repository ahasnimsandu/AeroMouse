import SwiftUI

private struct PercentilePoint: Identifiable {
    let id: String
    let q: Double
}

private struct LegendItem: Identifiable {
    let id: String
    let label: String
    let color: Color
}

private let percentilePoints: [PercentilePoint] = [
    PercentilePoint(id: "p50", q: 0.50),
    PercentilePoint(id: "p75", q: 0.75),
    PercentilePoint(id: "p90", q: 0.90),
    PercentilePoint(id: "p95", q: 0.95),
    PercentilePoint(id: "max", q: 1.0),
]

private let legendItems: [LegendItem] = [
    LegendItem(id: "good", label: "<3 px", color: .green),
    LegendItem(id: "ok", label: "3–8 px", color: .yellow),
    LegendItem(id: "warn", label: "8–15 px", color: .orange),
    LegendItem(id: "bad", label: ">15 px", color: .red),
]

/// Detailed accuracy readout for a calibrated model: where the error is, which direction it leans,
/// how the errors are distributed, and how the tail looks before versus after the stack.
struct AccuracyChartView: View {
    let stats: CalibrationStats

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if stats.cells.isEmpty {
                Text("Calibrate to see the error map.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            } else {
                residualMap
                HStack(alignment: .top, spacing: 14) {
                    histogram
                    percentiles.frame(width: 158)
                }
                legend
            }
        }
    }

    // MARK: Residual map

    private var residualMap: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Error map — ring colour = median error, arrow = systematic bias")
                .font(.caption2)
                .foregroundStyle(.secondary)

            GeometryReader { geometry in
                let inset: CGFloat = 10
                let rect = CGRect(x: inset, y: inset,
                                  width: max(geometry.size.width - inset * 2, 1),
                                  height: max(geometry.size.height - inset * 2, 1))

                Canvas { context, _ in
                    context.stroke(Path(roundedRect: rect, cornerRadius: 6),
                                   with: .color(.secondary.opacity(0.25)),
                                   style: StrokeStyle(lineWidth: 1, dash: [3, 3]))

                    for cell in stats.cells {
                        let center = CGPoint(x: rect.minX + CGFloat(cell.target.x) * rect.width,
                                             y: rect.minY + CGFloat(cell.target.y) * rect.height)
                        let radius: CGFloat = 5
                        let circle = Path(ellipseIn: CGRect(x: center.x - radius, y: center.y - radius,
                                                            width: radius * 2, height: radius * 2))
                        context.fill(circle, with: .color(ringColor(cell.medianAfter).opacity(0.78)))

                        // Bias arrow, magnified 6x and capped so it stays readable.
                        var dx = CGFloat(cell.bias.x) * 6
                        var dy = CGFloat(cell.bias.y) * 6
                        let limit: CGFloat = 16
                        let length = (dx * dx + dy * dy).squareRoot()
                        if length > limit {
                            dx *= limit / length
                            dy *= limit / length
                        }
                        if (dx * dx + dy * dy).squareRoot() > 1.5 {
                            var arrow = Path()
                            arrow.move(to: center)
                            arrow.addLine(to: CGPoint(x: center.x + dx, y: center.y + dy))
                            context.stroke(arrow, with: .color(.primary.opacity(0.5)),
                                           style: StrokeStyle(lineWidth: 1, lineCap: .round))
                        }
                    }
                }
            }
            .frame(height: 168)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.primary.opacity(0.03))
            )
        }
    }

    private func ringColor(_ error: Double) -> Color {
        switch error {
        case ..<3: return .green
        case ..<8: return .yellow
        case ..<15: return .orange
        default: return .red
        }
    }

    // MARK: Histogram

    private var histogram: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Error distribution (held-out samples)")
                .font(.caption2)
                .foregroundStyle(.secondary)

            Canvas { context, size in
                let bins = 14
                let maxError = max(stats.errorsBefore.max() ?? 1, stats.errorsAfter.max() ?? 1, 1)
                let width = size.width
                let height = size.height - 14

                var beforeBins = [Int](repeating: 0, count: bins)
                var afterBins = [Int](repeating: 0, count: bins)
                for error in stats.errorsBefore {
                    beforeBins[min(bins - 1, Int(error / maxError * Double(bins)))] += 1
                }
                for error in stats.errorsAfter {
                    afterBins[min(bins - 1, Int(error / maxError * Double(bins)))] += 1
                }
                let peak = max(beforeBins.max() ?? 1, afterBins.max() ?? 1, 1)
                let slot = width / CGFloat(bins)

                for index in 0..<bins {
                    let beforeHeight = CGFloat(beforeBins[index]) / CGFloat(peak) * height
                    let afterHeight = CGFloat(afterBins[index]) / CGFloat(peak) * height
                    let x = slot * CGFloat(index)

                    let beforeRect = CGRect(x: x + 1, y: height - beforeHeight,
                                            width: slot - 2, height: max(beforeHeight, 0.5))
                    context.fill(Path(roundedRect: beforeRect, cornerRadius: 1.5),
                                 with: .color(.secondary.opacity(0.22)))
                    context.stroke(Path(roundedRect: beforeRect, cornerRadius: 1.5),
                                   with: .color(.secondary.opacity(0.55)), lineWidth: 1)

                    let afterRect = CGRect(x: x + slot * 0.22, y: height - afterHeight,
                                           width: slot * 0.56, height: max(afterHeight, 0.5))
                    context.fill(Path(roundedRect: afterRect, cornerRadius: 1.5),
                                 with: .color(.green.opacity(0.75)))
                }

                context.stroke(Path { path in
                    path.move(to: CGPoint(x: 0, y: height))
                    path.addLine(to: CGPoint(x: width, y: height))
                }, with: .color(.secondary.opacity(0.35)), lineWidth: 1)

                let label = Text(String(format: "0–%.0f px", maxError))
                    .font(.system(size: 9))
                    .foregroundColor(.secondary)
                context.draw(label, at: CGPoint(x: width - 26, y: size.height - 5), anchor: .center)
            }
            .frame(height: 108)

            HStack(spacing: 10) {
                legendChip(color: .secondary.opacity(0.35), label: "before")
                legendChip(color: .green.opacity(0.75), label: "after")
            }
        }
    }

    private func legendChip(color: Color, label: String) -> some View {
        HStack(spacing: 3) {
            RoundedRectangle(cornerRadius: 2).fill(color).frame(width: 9, height: 9)
            Text(label).font(.system(size: 9)).foregroundStyle(.secondary)
        }
    }

    // MARK: Percentiles

    private var percentiles: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text("Percentiles (px)")
                .font(.caption2)
                .foregroundStyle(.secondary)

            let peak = max(percentile(stats.errorsBefore, 0.95),
                           percentile(stats.errorsAfter, 0.95), 1)

            ForEach(percentilePoints) { point in
                let before = percentile(stats.errorsBefore, point.q)
                let after = percentile(stats.errorsAfter, point.q)
                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Text(point.id).font(.system(size: 9)).foregroundStyle(.secondary)
                        Spacer()
                        Text(String(format: "%.1f → %.1f", before, after))
                            .font(.system(size: 9).monospacedDigit())
                            .foregroundStyle(after < before ? Color.green : Color.secondary)
                    }
                    GeometryReader { geometry in
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 1.5)
                                .fill(Color.secondary.opacity(0.18))
                                .frame(width: max(geometry.size.width * CGFloat(min(before / peak, 1)), 1),
                                       height: 3)
                            RoundedRectangle(cornerRadius: 1.5)
                                .fill(Color.green.opacity(0.8))
                                .frame(width: max(geometry.size.width * CGFloat(min(after / peak, 1)), 1),
                                       height: 3)
                        }
                    }
                    .frame(height: 3)
                }
            }
        }
    }

    private var legend: some View {
        HStack(spacing: 12) {
            ForEach(legendItems) { item in
                HStack(spacing: 4) {
                    Circle().fill(item.color).frame(width: 7, height: 7)
                    Text(item.label).font(.system(size: 9)).foregroundStyle(.secondary)
                }
            }
            Spacer()
            Text("\(stats.cells.count) cells · \(stats.errorsAfter.count) samples")
                .font(.system(size: 9))
                .foregroundStyle(.tertiary)
        }
    }

    // MARK: Math

    private func percentile(_ values: [Double], _ q: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let position = q * Double(sorted.count - 1)
        let lower = Int(position.rounded(.down))
        let upper = min(lower + 1, sorted.count - 1)
        let fraction = position - Double(lower)
        return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
    }
}

/// Small live trace of residual jitter, so smoothness is visible while you move your hand.
struct JitterSparkline: View {
    let values: [Double]

    var body: some View {
        Canvas { context, size in
            let maxValue = max(values.max() ?? 0.5, 0.5)
            if values.count > 1 {
                var path = Path()
                for (index, value) in values.enumerated() {
                    let x = size.width * CGFloat(index) / CGFloat(values.count - 1)
                    let y = size.height * (1 - CGFloat(min(value / maxValue, 1)))
                    if index == 0 {
                        path.move(to: CGPoint(x: x, y: y))
                    } else {
                        path.addLine(to: CGPoint(x: x, y: y))
                    }
                }
                context.stroke(path, with: .color(.green.opacity(0.85)),
                               style: StrokeStyle(lineWidth: 1.4, lineCap: .round, lineJoin: .round))
            }
            context.stroke(Path { path in
                path.move(to: CGPoint(x: 0, y: size.height - 0.5))
                path.addLine(to: CGPoint(x: size.width, y: size.height - 0.5))
            }, with: .color(.secondary.opacity(0.2)), lineWidth: 0.5)
        }
        .frame(height: 30)
    }
}
