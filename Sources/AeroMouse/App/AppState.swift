import Foundation
import SwiftUI
import AppKit
import AVFoundation

enum CalibrationPhase: Equatable {
    case idle
    case collecting
    case training
    case finished
    case failed(String)
}

/// One click sets the whole pointer feel: smoothing, prediction, precision gain and deadband move
/// together, because tuning them independently is what makes a pointer feel wrong in ways that are
/// hard to describe.
enum SmoothingPreset: String, CaseIterable, Identifiable {
    case precise, balanced, snappy

    var id: String { rawValue }

    var title: String {
        switch self {
        case .precise: return "Precise"
        case .balanced: return "Balanced"
        case .snappy: return "Snappy"
        }
    }

    var detail: String {
        switch self {
        case .precise: return "Heavy smoothing, small movements stay slow. Best for detail work."
        case .balanced: return "The default balance of smooth and responsive."
        case .snappy: return "Light smoothing with more prediction. Best for big screens."
        }
    }
}

@MainActor
final class AppState: ObservableObject {
    @Published var settings: AppSettings {
        didSet {
            engine.update(settings: settings)
            ModelStore.saveSettings(settings)
        }
    }
    @Published private(set) var model: PointerModel?
    @Published var snapshot = TrackingSnapshot()
    @Published var isTracking = false
    @Published var statusText = "Idle"
    @Published var cameraAuthorized = false
    @Published var accessibilityTrusted = false

    @Published var calibrationPhase: CalibrationPhase = .idle
    @Published var calibrationTargets: [Vec2] = []
    @Published var calibrationIndex = 0
    @Published var calibrationSampleCount = 0
    @Published var calibrationPlan = CalibrationPlan.standard()
    @Published var calibrationReport: CalibrationStats?
    @Published var selfTestResult: String = ""
    @Published var jitterHistory: [Double] = []

    let engine = TrackingEngine()
    let calibrationWindow = CalibrationWindowController()

    private var samples: [CalibrationSample] = []
    private var trainingTask: Task<Void, Never>?

    init() {
        settings = ModelStore.loadSettings()
        model = ModelStore.loadModel()
        engine.update(settings: settings)
        engine.update(model: model)

        engine.onSnapshot = { [weak self] snapshot in
            guard let self else { return }
            self.snapshot = snapshot
            self.isTracking = snapshot.isRunning
            if let suggestion = snapshot.tapSuggestion {
                self.settings.gestures.tapOn = suggestion.onLevel
                self.settings.gestures.tapOff = suggestion.offLevel
                self.settings.gestures.tapBaseline = suggestion.baseline
                self.settings.gestures.tapAdaptiveThreshold = true
                self.settings.tapTuned = true
                self.statusText = String(format: "Tap tuned: closed below %.3f, released above %.3f",
                                         suggestion.onLevel, suggestion.offLevel)
                self.engine.clearTapSuggestion()
            }
            if snapshot.isRunning {
                self.jitterHistory.append(snapshot.jitter)
                if self.jitterHistory.count > 150 {
                    self.jitterHistory.removeFirst(self.jitterHistory.count - 150)
                }
            } else if !self.jitterHistory.isEmpty {
                self.jitterHistory.removeAll(keepingCapacity: true)
            }
        }
        // Delivered from the camera queue, so hop to the main actor before touching published state.
        engine.onCalibrationSample = { [weak self] sample in
            Task { @MainActor [weak self] in
                self?.handleCalibrationSample(sample)
            }
        }

        refreshPermissions()
        updateScreen()

        if !UserDefaults.standard.bool(forKey: "hasSeenIntro") {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) { [weak self] in
                guard let self else { return }
                IntroWindowController.shared.show(app: self)
            }
        }
    }

    // MARK: Permissions

    func refreshPermissions() {
        cameraAuthorized = Permissions.cameraAuthorized
        accessibilityTrusted = Permissions.accessibilityTrusted
    }

    func requestCamera() {
        Permissions.requestCamera { [weak self] granted in
            guard let self else { return }
            self.cameraAuthorized = granted
            self.statusText = granted ? "Camera ready" : "Camera access denied"
        }
    }

    func requestAccessibility() {
        Permissions.promptAccessibility()
        accessibilityTrusted = Permissions.accessibilityTrusted
        statusText = "Grant Accessibility, then relaunch AeroMouse"
    }

    // MARK: Tracking

    func startTracking() {
        refreshPermissions()
        guard cameraAuthorized else {
            requestCamera()
            statusText = "Waiting for camera permission"
            return
        }
        guard accessibilityTrusted else {
            requestAccessibility()
            return
        }
        updateScreen()
        if let error = engine.start() {
            statusText = error
            return
        }
        isTracking = true
        statusText = model == nil ? "Tracking — uncalibrated rough mapping" : "Tracking"
    }

    func stopTracking() {
        engine.stop()
        isTracking = false
        statusText = "Paused"
    }

    func toggleTracking() {
        if isTracking { stopTracking() } else { startTracking() }
    }

    func updateScreen() {
        // The pointer has to be mapped onto the display it is actually on. Using the main screen
        // put every movement on the wrong monitor for anyone with more than one.
        let location = NSEvent.mouseLocation
        let screen = NSScreen.screens.first { NSMouseInRect(location, $0.frame, false) }
            ?? NSScreen.main
            ?? NSScreen.screens.first
        guard let rect = AppState.cgRect(for: screen) else { return }
        engine.setScreen(rect)
    }

    // MARK: Feel

    func apply(preset: SmoothingPreset) {
        var smoothing = settings.smoothing
        switch preset {
        case .precise:
            smoothing.minCutoff = 1.2
            smoothing.beta = 0.018
            smoothing.predictionMs = 12
            smoothing.precisionGain = 0.48
            smoothing.deadband = 1.3
        case .balanced:
            smoothing.minCutoff = 1.6
            smoothing.beta = 0.028
            smoothing.predictionMs = 16
            smoothing.precisionGain = 0.65
            smoothing.deadband = 0.9
        case .snappy:
            smoothing.minCutoff = 2.8
            smoothing.beta = 0.055
            smoothing.predictionMs = 20
            smoothing.precisionGain = 0.80
            smoothing.deadband = 0.7
        }
        settings.smoothing = smoothing
        statusText = "Feel set to \(preset.title)"
    }

    // MARK: Tap training

    func beginTapTraining() {
        refreshPermissions()
        if !isTracking { startTracking() }
        engine.beginTapTraining()
        statusText = "Relax your hand, then tap three times"
    }

    func cancelTapTraining() {
        engine.cancelTapTraining()
        statusText = "Tap training cancelled"
    }

    static func cgRect(for screen: NSScreen?) -> RectD? {
        guard let screen else { return nil }
        // NSScreen uses bottom-left origin; CGEvent coordinates are top-left of the primary display.
        let frame = screen.frame
        let primaryHeight = NSScreen.screens.first?.frame.height ?? frame.height
        let cg = CGRect(x: frame.origin.x,
                        y: primaryHeight - frame.maxY,
                        width: frame.width,
                        height: frame.height)
        return RectD(cg)
    }

    // MARK: Calibration

    var currentCalibrationTarget: Vec2? {
        guard calibrationIndex >= 0, calibrationIndex < calibrationTargets.count else { return nil }
        return calibrationTargets[calibrationIndex]
    }

    var calibrationProgress: Double {
        guard !calibrationTargets.isEmpty else { return 0 }
        return Double(min(calibrationIndex, calibrationTargets.count)) / Double(calibrationTargets.count)
    }

    func startCalibration(plan: CalibrationPlan? = nil) {
        if let plan { calibrationPlan = plan }
        samples.removeAll()
        calibrationSampleCount = 0
        calibrationReport = nil
        calibrationTargets = calibrationPlan.targets()
        calibrationIndex = 0
        calibrationPhase = .collecting
        updateScreen()

        if !isTracking {
            startTracking()
        }
        engine.setCalibration(target: currentCalibrationTarget, active: true)
        calibrationWindow.show(with: self, screen: NSScreen.main ?? NSScreen.screens.first)
    }

    func handleCalibrationSample(_ sample: CalibrationSample) {
        guard calibrationPhase == .collecting, let target = currentCalibrationTarget else { return }
        var recorded = sample
        recorded.target = target
        recorded.group = calibrationPlan.groupIndex(for: calibrationIndex)
        samples.append(recorded)
        calibrationSampleCount = samples.count
        engine.noteSampleRecorded()

        // A pinch is deliberate: advance to the next target. Hover samples just accumulate.
        if sample.weight >= 1.0 {
            advanceCalibration()
        }
    }

    func skipCalibrationTarget() {
        guard calibrationPhase == .collecting else { return }
        advanceCalibration()
    }

    private func advanceCalibration() {
        calibrationIndex += 1
        if calibrationIndex >= calibrationTargets.count {
            finishCalibration()
        } else {
            engine.setCalibration(target: currentCalibrationTarget, active: true)
        }
    }

    private func finishCalibration() {
        engine.setCalibration(target: nil, active: false)
        let collected = samples
        let screen = engine.currentScreenRect()
        calibrationPhase = .training

        guard collected.count >= 16 else {
            calibrationPhase = .failed("Only \(collected.count) samples collected. Keep your hand in frame and pinch each dot.")
            return
        }

        trainingTask?.cancel()
        trainingTask = Task { [weak self] in
            // Cross-validating four mappings plus a neural net takes a moment; keep it off the
            // main actor so the UI stays responsive.
            let built = await Task.detached(priority: .userInitiated) {
                Calibrator.build(samples: collected, screen: screen)
            }.value
            guard let self else { return }
            self.applyCalibrationResult(built)
        }
    }

    private func applyCalibrationResult(_ built: PointerModel?) {
        guard let built else {
            calibrationPhase = .failed("Could not fit a stable mapping. Try again with your hand fully in frame and every dot pinched.")
            return
        }
        model = built
        ModelStore.saveModel(built)
        engine.update(model: built)
        calibrationReport = built.stats
        calibrationPhase = .finished
        statusText = String(format: "Calibrated — median error %.1f px", built.stats.medianAfter)
    }

    func cancelCalibration() {
        trainingTask?.cancel()
        engine.setCalibration(target: nil, active: false)
        calibrationPhase = .idle
        calibrationWindow.close()
    }

    func dismissCalibrationSummary() {
        calibrationPhase = .idle
        calibrationWindow.close()
    }

    func resetModel() {
        ModelStore.deleteModel()
        model = nil
        calibrationReport = nil
        engine.update(model: nil)
        statusText = "Model deleted — calibrate again"
    }

    // MARK: Diagnostics

    func runSelfTest() {
        selfTestResult = "Running…"
        Task.detached(priority: .userInitiated) {
            let summary = SelfTest.summary()
            await MainActor.run {
                self.selfTestResult = summary.failures == 0
                    ? "\(summary.checks)/\(summary.checks) checks passed"
                    : "\(summary.failures) of \(summary.checks) checks failed"
            }
        }
    }

    func openModelFolder() {
        ModelStore.ensureDirectory()
        NSWorkspace.shared.open(ModelStore.directory)
    }
}
