import SwiftUI

struct CalibrationOverlayView: View {
    @EnvironmentObject var app: AppState

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                Color.black.opacity(0.9).ignoresSafeArea()

                switch app.calibrationPhase {
                case .collecting, .training:
                    collecting(size: geometry.size)
                case .finished:
                    summary
                case .failed(let message):
                    failure(message)
                case .idle:
                    Text("Nothing to calibrate")
                        .foregroundStyle(.secondary)
                }
            }
        }
        .onExitCommand { app.cancelCalibration() }
    }

    // MARK: Collecting

    private func collecting(size: CGSize) -> some View {
        let target = app.currentCalibrationTarget ?? Vec2(0.5, 0.5)
        let point = CGPoint(x: target.x * size.width, y: target.y * size.height)

        return ZStack {
            ForEach(1..<4) { ring in
                Circle()
                    .stroke(Color.white.opacity(0.30 / Double(ring)), lineWidth: 1.5)
                    .frame(width: 24 + CGFloat(ring) * 26, height: 24 + CGFloat(ring) * 26)
                    .position(point)
            }
            Circle()
                .fill(Color.white.opacity(0.92))
                .frame(width: 10, height: 10)
                .position(point)
            Circle()
                .fill(Color.accentColor)
                .frame(width: 4, height: 4)
                .position(point)

            VStack(spacing: 10) {
                Text("Calibration")
                    .font(.title2.weight(.semibold))
                Text("Move the pointer with your hand, then blink both eyes on the dot.\nVary your distance and hand angle slightly — the fit uses that to learn depth and roll.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)

                if !app.snapshot.hasHand {
                    Label("Hand not detected — raise it into the camera view", systemImage: "exclamationmark.triangle")
                        .font(.callout)
                        .foregroundStyle(.orange)
                        .padding(.top, 4)
                }

                Spacer()

                HStack(spacing: 18) {
                    Text("Dot \(min(app.calibrationIndex + 1, app.calibrationTargets.count)) of \(app.calibrationTargets.count)")
                    Text("•")
                    Text("\(app.calibrationSampleCount) samples")
                    if app.snapshot.hasFace {
                        Text("•")
                        Text(String(format: "eyes %.0f%%", app.snapshot.gazeOpenness * 100))
                            .foregroundStyle(app.snapshot.gazeOpenness > 0.30 ? .green : .orange)
                    }
                }
                .font(.callout.monospacedDigit())
                .foregroundStyle(.secondary)

                ProgressView(value: app.calibrationProgress)
                    .frame(width: 340)

                HStack(spacing: 12) {
                    Button("Skip dot") { app.skipCalibrationTarget() }
                        .disabled(app.calibrationPhase != .collecting)
                    Button("Cancel") { app.cancelCalibration() }
                        .keyboardShortcut(.cancelAction)
                }
                .padding(.top, 4)

                if app.calibrationPhase == .training {
                    ProgressView("Training model — cross-validating every mapping…")
                        .padding(.top, 6)
                }
            }
            .padding(30)
        }
    }

    // MARK: Result

    private var summary: some View {
        VStack(spacing: 16) {
            Image(systemName: "target")
                .font(.system(size: 34))
                .foregroundStyle(.green)
            Text("Calibration complete").font(.title2.weight(.semibold))

            if let stats = app.calibrationReport {
                VStack(spacing: 8) {
                    HStack(spacing: 28) {
                        statBlock("Median before", String(format: "%.1f px", stats.medianBefore))
                        statBlock("Median after", String(format: "%.1f px", stats.medianAfter), highlight: true)
                        statBlock("p95 after", String(format: "%.1f px", stats.p95After))
                    }
                    Divider().frame(width: 420)
                    Text("\(stats.samples) samples across \(stats.targets) target cells")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                    Text("Layers: " + stats.chosen.joined(separator: ", "))
                        .font(.callout)
                        .foregroundStyle(.secondary)
                    Text(stats.residualNetIncluded
                         ? "Neural residual correction trained on your hand."
                         : "Neural correction skipped — the geometric stack was already better under cross-validation.")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
                .padding(18)
                .background(RoundedRectangle(cornerRadius: 12).fill(.white.opacity(0.06)))
            }

            Button("Done") { app.dismissCalibrationSummary() }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
        }
        .padding(30)
    }

    private func statBlock(_ title: String, _ value: String, highlight: Bool = false) -> some View {
        VStack(spacing: 3) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value)
                .font(.title3.monospacedDigit())
                .foregroundStyle(highlight ? Color.green : Color.primary)
        }
    }

    // MARK: Failure

    private func failure(_ message: String) -> some View {
        VStack(spacing: 14) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 30))
                .foregroundStyle(.orange)
            Text("Calibration failed").font(.title3.weight(.semibold))
            Text(message)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 460)
            HStack(spacing: 12) {
                Button("Try again") {
                    app.cancelCalibration()
                    app.startCalibration()
                }
                Button("Close") { app.cancelCalibration() }
                    .keyboardShortcut(.cancelAction)
            }
        }
        .padding(30)
    }
}
