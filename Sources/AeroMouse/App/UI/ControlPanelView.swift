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

struct SliderRow: View {
    let label: String
    let value: Binding<Double>
    let range: ClosedRange<Double>
    let format: String
    var hint: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(label).font(.callout)
                Spacer()
                Text(String(format: format, value.wrappedValue))
                    .font(.callout.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Slider(value: value, in: range)
            if let hint {
                Text(hint).font(.caption2).foregroundStyle(.tertiary)
            }
        }
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

// MARK: - Control panel

struct ControlPanelView: View {
    @EnvironmentObject var app: AppState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                HStack(alignment: .top, spacing: 16) {
                    VStack(spacing: 16) {
                        trackerCard
                        liveMetricsCard
                        lowLightCard
                        eyeCard
                    }
                    .frame(maxWidth: 360)
                    VStack(spacing: 16) {
                        accuracyCard
                        smoothingCard
                        gesturesCard
                        permissionsCard
                        diagnosticsCard
                    }
                }
            }
            .padding(18)
        }
        .frame(minWidth: 860, minHeight: 760)
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
        }
    }

    // MARK: Tracker

    private var trackerCard: some View {
        Card(title: "Camera", subtitle: "Green skeleton is what Vision sees; the red ring is the pointer anchor.") {
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

    // MARK: Live metrics

    private var liveMetricsCard: some View {
        Card(title: "Live") {
            MetricRow(label: "Camera", value: cameraText)
            MetricRow(label: "Vision per frame", value: String(format: "%.1f ms", app.snapshot.visionMilliseconds))
            MetricRow(label: "Residual jitter",
                      value: String(format: "%.2f px", app.snapshot.jitter),
                      tint: app.snapshot.jitter < 0.6 ? .green : .orange)
            MetricRow(label: "Thumb→index", value: pinchText(app.snapshot.pinchIndex))
            MetricRow(label: "Thumb→middle", value: pinchText(app.snapshot.pinchMiddle))
            MetricRow(label: "Tap contact",
                      value: app.snapshot.thumbContact.isFinite
                        ? String(format: "%.3f", app.snapshot.thumbContact) : "—",
                      tint: tapClosed ? .green : .secondary)
            MetricRow(label: "Fist", value: String(format: "%.2f", app.snapshot.fist))
            MetricRow(label: "Light", value: String(format: "%.0f%%", app.snapshot.luma * 100),
                      tint: app.snapshot.luma < 0.25 ? .orange : .secondary)
            MetricRow(label: "Tracking quality",
                      value: String(format: "%.0f%%", app.snapshot.quality * 100),
                      tint: app.snapshot.quality > 0.75 ? .green : .orange)
            MetricRow(label: "Pipeline",
                      value: app.snapshot.enhancing ? "ANE + GPU low-light" : "ANE hand pose",
                      tint: .secondary)
            MetricRow(label: "Finger flow",
                      value: app.snapshot.flowLocked ? "locked (sub-pixel)" : "landmark only",
                      tint: app.snapshot.flowLocked ? .green : .secondary)
            VStack(alignment: .leading, spacing: 2) {
                Text("Jitter trace").font(.caption2).foregroundStyle(.secondary)
                JitterSparkline(values: app.jitterHistory)
            }
            HStack(spacing: 6) {
                if app.snapshot.dragging { StatusPill(text: "Holding", active: true) }
                if app.snapshot.frozen { StatusPill(text: "Frozen", active: false) }
                if app.snapshot.hasFace { StatusPill(text: "Face", active: true) }
            }
        }
    }

    private var tapClosed: Bool {
        // Use the detector's live level, not the configured ceiling: they differ once the resting
        // gap is being learned, and the panel should show what the recogniser is actually doing.
        let snapshot = app.snapshot
        guard snapshot.thumbContact.isFinite else { return false }
        return snapshot.tapOnLevel > 0
            ? snapshot.thumbContact < snapshot.tapOnLevel
            : snapshot.thumbContact < app.settings.gestures.tapOn
    }

    /// Resolution matters more than frame rate for pointing accuracy: twice the pixels halves the
    /// landmark error, and that error is multiplied again on the way to the screen.
    private var cameraText: String {
        let snapshot = app.snapshot
        let resolution = snapshot.cameraWidth > 0
            ? "\(snapshot.cameraWidth)×\(snapshot.cameraHeight)"
            : "—"
        return String(format: "%@ · %.0f fps", resolution, snapshot.fps)
    }

    private var lowLightCard: some View {
        Card(title: "Low light & GPU",
             subtitle: "Hand pose runs on the Neural Engine. Dim frames get GPU tone mapping, temporal averaging and a longer camera exposure.") {
            Toggle("HD capture (sharper landmarks, restart to apply)",
                   isOn: $app.settings.preferHighResolution)
                .toggleStyle(.switch)
                .controlSize(.small)
            Toggle("Enhance dark frames (GPU)", isOn: $app.settings.lowLightEnhancement)
                .toggleStyle(.switch)
                .controlSize(.small)
            Toggle("Adaptive frame rate", isOn: $app.settings.adaptiveFrameRate)
                .toggleStyle(.switch)
                .controlSize(.small)
            Toggle("Sub-pixel finger tracking (optical flow)", isOn: $app.settings.fingerFlowEnabled)
                .toggleStyle(.switch)
                .controlSize(.small)
            Text("Dropping to 30 fps doubles the camera's exposure time, which removes sensor noise before tracking instead of amplifying it afterwards. All 21 landmarks are also filtered and held briefly through dropouts.")
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func pinchText(_ value: Double) -> String {
        value.isFinite ? String(format: "%.2f", value) : "—"
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
                    MetricRow(label: "Residual spline stage",
                              value: stats.residualSplineIncluded ? "active" : "not needed",
                              tint: stats.residualSplineIncluded ? .green : .secondary)
                    MetricRow(label: "Neural correction",
                              value: stats.residualNetIncluded ? "active" : "not needed",
                              tint: stats.residualNetIncluded ? .green : .secondary)
                    if let gaze = stats.gazeRMS {
                        MetricRow(label: "Gaze assist", value: String(format: "±%.0f px", gaze))
                    }
                }

                Divider()
                AccuracyChartView(stats: stats)
            } else {
                Text("No model yet. Calibrate to fit the projective map, polynomial regression, thin-plate spline and the neural residual net, then pick whichever wins cross-validation.")
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

                Button("Reset model") { app.resetModel() }
                    .disabled(app.model == nil)
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

    // MARK: Smoothing

    private var smoothingCard: some View {
        Card(title: "Smoothing", subtitle: "One Euro filter with velocity prediction. Lower cutoff = silkier; higher = snappier.") {
            SliderRow(label: "Smoothing (min cutoff)", value: $app.settings.smoothing.minCutoff,
                      range: 0.4...6.0, format: "%.1f Hz",
                      hint: "Lower is smoother but adds lag. 1.5–2.5 feels best for most hands.")
            SliderRow(label: "Speed responsiveness", value: $app.settings.smoothing.beta,
                      range: 0.0...0.20, format: "%.3f",
                      hint: "How quickly fast movement escapes the filter.")
            SliderRow(label: "Prediction", value: $app.settings.smoothing.predictionMs,
                      range: 0...40, format: "%.0f ms",
                      hint: "Compensates the filter's latency. Too high overshoots.")
            SliderRow(label: "Precision gain", value: $app.settings.smoothing.precisionGain,
                      range: 0.30...1.0, format: "%.2f",
                      hint: "Movement scale while moving slowly, for hitting small targets.")
            SliderRow(label: "Deadband", value: $app.settings.smoothing.deadband,
                      range: 0...4, format: "%.1f px",
                      hint: "Stops one-pixel shimmer while you hold still.")
            SliderRow(label: "Jump rejection", value: $app.settings.smoothing.outlierGate,
                      range: 40...400, format: "%.0f px",
                      hint: "Frames that teleport further than this are ignored.")
        }
    }

    // MARK: Gestures

    private var gesturesCard: some View {
        Card(title: "Gestures",
             subtitle: "Tap the thumb against the side of the index finger to click, the way double tap works on Apple Watch. Keep the contact and move to drag.") {
            HStack(spacing: 22) {
                Toggle("Tap to click (thumb taps index)", isOn: $app.settings.gestures.tapEnabled)
                Toggle("Pinch to click", isOn: $app.settings.gestures.pinchClickEnabled)
            }
            .toggleStyle(.switch)
            .controlSize(.small)
            Toggle("Learn my resting gap automatically",
                   isOn: $app.settings.gestures.tapAdaptiveThreshold)
                .toggleStyle(.switch)
                .controlSize(.small)
            HStack(spacing: 10) {
                Button(app.snapshot.tapTrainingActive ? "Cancel" : "Learn my tap…") {
                    if app.snapshot.tapTrainingActive { app.cancelTapTraining() }
                    else { app.beginTapTraining() }
                }
                if app.snapshot.tapTrainingActive {
                    ProgressView(value: app.snapshot.tapTrainingProgress)
                        .frame(width: 130)
                }
                Spacer()
            }
            MetricRow(label: "Resting gap", value: String(format: "%.3f", app.snapshot.contactBaseline))
            MetricRow(label: "Fires below", value: String(format: "%.3f", app.snapshot.tapOnLevel),
                      tint: app.snapshot.tapClosed ? .green : .secondary)
            MetricRow(label: "Releases above", value: String(format: "%.3f", app.snapshot.tapOffLevel))
            SliderRow(label: "Tap contact ceiling", value: $app.settings.gestures.tapOn,
                      range: 0.03...0.20, format: "%.3f",
                      hint: "A click can never fire above this gap. The live threshold is this, or a real drop below your resting gap, whichever is smaller — that is what stops normal hand movement from clicking.")
            SliderRow(label: "Tap release threshold", value: $app.settings.gestures.tapOff,
                      range: 0.05...0.30, format: "%.3f",
                      hint: "Keep this above the contact threshold — that gap is what stops a resting thumb from clicking twice.")
            SliderRow(label: "Pinch close threshold", value: $app.settings.gestures.pinchOn,
                      range: 0.20...0.80, format: "%.2f")
                .disabled(!app.settings.gestures.pinchClickEnabled)
            SliderRow(label: "Pinch release threshold", value: $app.settings.gestures.pinchOff,
                      range: 0.30...1.00, format: "%.2f",
                      hint: "Always keep release above close: that gap is the anti-flicker hysteresis.")
                .disabled(!app.settings.gestures.pinchClickEnabled)
            HStack(spacing: 22) {
                Toggle("Right click (thumb + middle)", isOn: $app.settings.gestures.rightClickEnabled)
                Toggle("Double click", isOn: $app.settings.gestures.doubleClickEnabled)
                Toggle("Fist freezes", isOn: $app.settings.gestures.freezeEnabled)
            }
            .toggleStyle(.switch)
            .controlSize(.small)
        }
    }

    // MARK: Eyes

    private var eyeCard: some View {
        Card(title: "Gaze assist",
             subtitle: "Both pupils, head angle and head position are regressed onto the screen and cross-validated over your calibration targets. It acts as a precise absolute prior: it pulls the pointer onto what you are looking at while your hand settles, and steps out of the way while you move.") {
            Toggle("Enable gaze assist", isOn: $app.settings.eyeAssistEnabled)
                .toggleStyle(.switch)
            SliderRow(label: "Strength", value: $app.settings.eyeAssistStrength,
                      range: 0.0...1.0, format: "%.2f",
                      hint: "How much of the gaze offset is applied. The clamp scales with the model's own accuracy, so a weak model can only nudge.")
                .disabled(!app.settings.eyeAssistEnabled)
            if let gaze = app.model?.stats.gazeRMS {
                MetricRow(label: "Model accuracy (held-out)",
                          value: String(format: "±%.0f px", gaze),
                          tint: gaze < 60 ? .green : .orange)
            } else {
                Text("Gaze trains itself from the calibration dots — keep your face in frame while you calibrate.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            if app.snapshot.hasFace {
                MetricRow(label: "Live correction",
                          value: app.snapshot.gazeActive ? "applying" : "waiting for the hand to settle",
                          tint: app.snapshot.gazeActive ? .green : .secondary)
                MetricRow(label: "Eye openness",
                          value: String(format: "%.0f%%", app.snapshot.gazeOpenness * 100))
            }
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
            if !app.accessibilityTrusted {
                Text("Moving the cursor and synthesising clicks requires Accessibility. After granting it, quit and reopen AeroMouse.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: Diagnostics

    private var diagnosticsCard: some View {
        Card(title: "Diagnostics") {
            HStack(spacing: 10) {
                Button("Run self-test") { app.runSelfTest() }
                Text(app.selfTestResult.isEmpty ? "Verifies geometry, regression, spline and the neural net." : app.selfTestResult)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Model folder") { app.openModelFolder() }
            }
        }
    }
}
