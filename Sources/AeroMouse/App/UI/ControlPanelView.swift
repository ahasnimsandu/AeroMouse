import SwiftUI
import AppKit

// MARK: - Building blocks

struct Card<Content: View>: View {
    let title: String
    var subtitle: String? = nil
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title.uppercased())
                    .font(.caption.weight(.semibold))
                    .tracking(0.6)
                    .foregroundStyle(.secondary)
                if let subtitle {
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            }
            content
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(nsColor: .controlBackgroundColor))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08))
        )
    }
}

struct MetricRow: View {
    let label: String
    let value: String
    var tint: Color = .primary

    var body: some View {
        HStack {
            Text(label).font(.callout).foregroundStyle(.secondary)
            Spacer()
            Text(value).font(.callout.monospacedDigit()).foregroundStyle(tint)
        }
    }
}

struct StatusPill: View {
    let text: String
    let active: Bool

    var body: some View {
        Text(text)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 9)
            .padding(.vertical, 4)
            .background(
                Capsule().fill(active ? Color.green.opacity(0.18) : Color.secondary.opacity(0.15))
            )
            .foregroundStyle(active ? .green : .secondary)
    }
}

struct GestureRow: View {
    let icon: String
    let title: String
    let detail: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 24, weight: .semibold))
                .foregroundStyle(.tint)
                .frame(width: 34)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.callout.weight(.semibold))
                Text(detail).font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
        }
    }
}

// MARK: - Control panel

struct ControlPanelView: View {
    @EnvironmentObject var app: AppState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                if !isSetUp { setupCard }
                HStack(alignment: .top, spacing: 16) {
                    VStack(spacing: 16) {
                        trackerCard
                        howToCard
                        liveCard
                    }
                    .frame(maxWidth: 380)
                    VStack(spacing: 16) {
                        accuracyCard
                        feelCard
                        permissionsCard
                        diagnosticsCard
                    }
                }
            }
            .padding(18)
        }
        .frame(minWidth: 880, minHeight: 700)
        .background(Color(nsColor: .windowBackgroundColor))
        .onAppear {
            app.refreshPermissions()
            app.updateScreen()
        }
    }

    // MARK: Header

    private var header: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text("AeroMouse").font(.title2.weight(.semibold))
                Text(app.statusText).font(.callout).foregroundStyle(.secondary)
            }
            Spacer()
            StatusPill(text: app.isTracking ? "Tracking" : "Paused", active: app.isTracking)
            StatusPill(text: app.model == nil ? "Uncalibrated" : "Calibrated",
                       active: app.model != nil)
            Button(app.isTracking ? "Pause" : "Start") {
                app.toggleTracking()
            }
            .keyboardShortcut(.space, modifiers: [.command])
            .buttonStyle(.borderedProminent)
            Button("Calibrate…") { app.startCalibration(plan: .standard()) }
                .buttonStyle(.bordered)
            Menu {
                Button("Quick calibration…") { app.startCalibration(plan: .quick()) }
                Button("Precise calibration…") { app.startCalibration(plan: .precise()) }
                Divider()
                Button("Welcome guide…") {
                    NSApp.activate(ignoringOtherApps: true)
                    IntroWindowController.shared.show(app: app)
                }
                Button("Run self-test") { app.runSelfTest() }
                Button("Open model folder") { app.openModelFolder() }
                Divider()
                Button("Delete calibration") { app.resetModel() }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
        }
    }

    // MARK: Setup

    private var isSetUp: Bool {
        app.cameraAuthorized && app.accessibilityTrusted && app.model != nil
    }

    private var setupCard: some View {
        Card(title: "Get started",
             subtitle: "Three steps and you are pointing with your hand.") {
            setupRow(done: app.cameraAuthorized,
                     title: "Allow camera access",
                     detail: "AeroMouse has to see your hand.",
                     actionTitle: "Allow",
                     action: { app.requestCamera() })
            setupRow(done: app.accessibilityTrusted,
                     title: "Allow Accessibility",
                     detail: "macOS only lets a trusted app move the pointer.",
                     actionTitle: "Open Settings",
                     action: { app.requestAccessibility() })
            setupRow(done: app.model != nil,
                     title: "Calibrate on your screen",
                     detail: "Follow the dots and pinch each one. Takes about a minute.",
                     actionTitle: "Calibrate",
                     action: { app.startCalibration(plan: .standard()) })
        }
    }

    private func setupRow(done: Bool, title: String, detail: String,
                          actionTitle: String, action: @escaping () -> Void) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Image(systemName: done ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(done ? Color.green : Color.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(title).font(.callout.weight(done ? .regular : .medium))
                Text(detail).font(.caption2).foregroundStyle(.tertiary)
            }
            Spacer()
            if !done {
                Button(actionTitle, action: action)
                    .controlSize(.small)
            }
        }
    }

    // MARK: Camera

    private var trackerCard: some View {
        Card(title: "Camera", subtitle: "Move your hand to move the pointer. Keep it inside the frame.") {
            ZStack {
                CameraPreview(session: app.engine.previewSession, mirrored: app.settings.mirrorPreview)
                HandOverlay(snapshot: app.snapshot, mirrored: app.settings.mirrorPreview)
                if !app.snapshot.hasHand {
                    VStack(spacing: 6) {
                        Image(systemName: "hand.raised.slash")
                            .font(.system(size: 22))
                        Text(app.isTracking ? "Show your hand to the camera" : "Tracking is paused")
                            .font(.caption)
                    }
                    .foregroundStyle(.white.opacity(0.75))
                    .padding(10)
                    .background(RoundedRectangle(cornerRadius: 8).fill(.black.opacity(0.45)))
                }
            }
            .aspectRatio(4.0 / 3.0, contentMode: .fit)
            .background(Color.black)
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))

            Toggle("Mirror preview", isOn: $app.settings.mirrorPreview)
                .toggleStyle(.switch)
                .controlSize(.small)
        }
    }

    // MARK: How to use

    private var howToCard: some View {
        Card(title: "Two gestures",
             subtitle: "That is all AeroMouse does. Nothing else fires by accident.") {
            GestureRow(icon: "hand.pinch.fill",
                       title: "Click",
                       detail: "Pinch your thumb and index finger together. Release to let go.")
            Divider()
            GestureRow(icon: "hand.point.up.left.fill",
                       title: "Right click",
                       detail: "Pinch your thumb and middle finger together. Keep the index finger clear.")
            Divider()
            Text("Live readings")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            MetricRow(label: "Thumb → index", value: pinchText(app.snapshot.pinchIndex),
                      tint: app.snapshot.pinchIndex < app.settings.gestures.pinchOn ? .green : .secondary)
            MetricRow(label: "Thumb → middle", value: pinchText(app.snapshot.pinchMiddle),
                      tint: app.snapshot.pinchMiddle < app.settings.gestures.middlePinchOn ? .green : .secondary)
        }
    }

    // MARK: Live

    private var liveCard: some View {
        Card(title: "Live") {
            MetricRow(label: "Camera", value: cameraText)
            MetricRow(label: "Vision per frame", value: String(format: "%.1f ms", app.snapshot.visionMilliseconds))
            MetricRow(label: "Jitter",
                      value: String(format: "%.2f px", app.snapshot.jitter),
                      tint: app.snapshot.jitter < 0.6 ? .green : .orange)
            MetricRow(label: "Tracking quality",
                      value: String(format: "%.0f%%", app.snapshot.quality * 100),
                      tint: app.snapshot.quality > 0.75 ? .green : .orange)
            VStack(alignment: .leading, spacing: 2) {
                Text("Jitter trace").font(.caption2).foregroundStyle(.secondary)
                JitterSparkline(values: app.jitterHistory)
            }
            HStack(spacing: 6) {
                if app.snapshot.dragging { StatusPill(text: "Holding", active: true) }
                if app.snapshot.tapClosed { StatusPill(text: "Left", active: true) }
            }
        }
    }

    private func pinchText(_ value: Double) -> String {
        value.isFinite ? String(format: "%.2f", value) : "—"
    }

    /// Resolution matters more than frame rate for pointing accuracy.
    private var cameraText: String {
        let snapshot = app.snapshot
        let resolution = snapshot.cameraWidth > 0
            ? "\(snapshot.cameraWidth)×\(snapshot.cameraHeight)"
            : "—"
        return String(format: "%@ · %.0f fps", resolution, snapshot.fps)
    }

    // MARK: Accuracy

    private var accuracyCard: some View {
        Card(title: "Accuracy",
             subtitle: app.model == nil
                ? "Uncalibrated: a rough mirror mapping is used until you calibrate."
                : "Measured by grouped cross-validation on held-out target cells.") {
            if let stats = app.model?.stats {
                VStack(spacing: 8) {
                    HStack(alignment: .firstTextBaseline, spacing: 14) {
                        accuracyColumn(title: "Best geometric", value: stats.medianBefore)
                        Image(systemName: "arrow.right").foregroundStyle(.tertiary)
                        accuracyColumn(title: "After stack", value: stats.medianAfter, highlight: true)
                        Spacer()
                        VStack(alignment: .trailing, spacing: 2) {
                            Text("p95").font(.caption).foregroundStyle(.secondary)
                            Text(String(format: "%.1f px", stats.p95After))
                                .font(.headline.monospacedDigit())
                        }
                    }

                    Divider()

                    MetricRow(label: "RMS (before → after)",
                              value: String(format: "%.1f → %.1f px", stats.rmsBefore, stats.rmsAfter))
                    MetricRow(label: "Samples / targets", value: "\(stats.samples) / \(stats.targets)")
                    MetricRow(label: "Fitted layers", value: stats.chosen.joined(separator: ", "))
                }

                Divider()
                AccuracyChartView(stats: stats)
            } else {
                Text("No model yet. Calibrate to fit the mapping for your screen, hand and camera.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Divider()

            HStack(spacing: 10) {
                Picker("", selection: Binding(
                    get: { app.calibrationPlan.columns },
                    set: { columns in
                        var plan = app.calibrationPlan
                        plan.columns = columns
                        plan.rows = columns
                        app.calibrationPlan = plan
                    }
                )) {
                    Text("Quick 3×3").tag(3)
                    Text("Standard 5×5").tag(5)
                    Text("Precise 7×7").tag(7)
                }
                .labelsHidden()
                .frame(width: 150)

                Stepper("Passes: \(app.calibrationPlan.passes)",
                        value: $app.calibrationPlan.passes, in: 1...6)
                    .font(.callout)

                Spacer()

                Button("Calibrate…") { app.startCalibration() }
                    .buttonStyle(.borderedProminent)
                    .disabled(app.calibrationPhase == .collecting || app.calibrationPhase == .training)
            }
        }
    }

    private func accuracyColumn(title: String, value: Double, highlight: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(String(format: "%.1f px", value))
                .font(highlight ? .title3.weight(.semibold).monospacedDigit() : .title3.monospacedDigit())
                .foregroundStyle(highlight ? Color.green : Color.primary)
        }
    }

    // MARK: Feel

    private var feelCard: some View {
        Card(title: "Feel",
             subtitle: "One click sets smoothing, prediction and precision together.") {
            HStack(spacing: 8) {
                ForEach(SmoothingPreset.allCases) { preset in
                    Button(preset.title) { app.apply(preset: preset) }
                        .buttonStyle(.bordered)
                        .controlSize(.large)
                        .help(preset.detail)
                }
                Spacer()
            }
            Text("Balanced is the best starting point. Precise is steadier for small targets; Snappy is quicker on large screens.")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: Permissions

    private var permissionsCard: some View {
        Card(title: "Permissions") {
            HStack {
                Circle().fill(app.cameraAuthorized ? Color.green : Color.orange).frame(width: 8, height: 8)
                Text("Camera").font(.callout)
                Spacer()
                if app.cameraAuthorized {
                    Text("Granted").font(.callout).foregroundStyle(.secondary)
                } else {
                    Button("Allow camera") { app.requestCamera() }
                }
            }
            HStack {
                Circle().fill(app.accessibilityTrusted ? Color.green : Color.orange).frame(width: 8, height: 8)
                Text("Accessibility").font(.callout)
                Spacer()
                if app.accessibilityTrusted {
                    Text("Granted").font(.callout).foregroundStyle(.secondary)
                } else {
                    Button("Open Settings") {
                        Permissions.promptAccessibility()
                        Permissions.openAccessibilitySettings()
                    }
                }
            }
        }
    }

    // MARK: Diagnostics

    private var diagnosticsCard: some View {
        Card(title: "Diagnostics") {
            HStack(spacing: 10) {
                Button("Run self-test") { app.runSelfTest() }
                Text(app.selfTestResult.isEmpty ? "Verifies the pointer and the gestures." : app.selfTestResult)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Model folder") { app.openModelFolder() }
            }
        }
    }
}
