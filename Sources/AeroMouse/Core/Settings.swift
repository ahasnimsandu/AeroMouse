import Foundation

// MARK: - App settings

struct AppSettings: Codable {
    var trackingEnabled: Bool = true
    var smoothing = SmoothingConfig()
    var gestures = GestureConfig()
    var eyeAssistEnabled: Bool = false
    var eyeAssistStrength: Double = 0.50
    var mirrorPreview: Bool = false
    var requestHighFrameRate: Bool = true
    var hoverSamplesEnabled: Bool = true
    var preferRightHand: Bool = true
    var lowLightEnhancement: Bool = true
    var adaptiveFrameRate: Bool = true
    var preferHighResolution: Bool = true
    var fingerFlowEnabled: Bool = true
    /// Set once the tap trainer has placed thresholds from this user's own hand.
    var tapTuned: Bool = false
}

// MARK: - Persistence

enum ModelStore {
    static var directory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        return base.appendingPathComponent("AeroMouse", isDirectory: true)
    }

    static var modelURL: URL { directory.appendingPathComponent("model.json") }
    static var settingsURL: URL { directory.appendingPathComponent("settings.json") }

    static func ensureDirectory() {
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    /// Bumped whenever the meaning of the model's inputs changes (anchor point, coordinate space).
    /// An older model would silently produce a wildly wrong mapping, so it is discarded instead.
    static let requiredModelVersion = 4

    static func loadModel() -> PointerModel? {
        guard let data = try? Data(contentsOf: modelURL),
              let model = try? JSONDecoder().decode(PointerModel.self, from: data),
              model.version >= requiredModelVersion else { return nil }
        return model
    }

    static func saveModel(_ model: PointerModel) {
        ensureDirectory()
        guard let data = try? JSONEncoder().encode(model) else { return }
        try? data.write(to: modelURL, options: .atomic)
    }

    static func deleteModel() {
        try? FileManager.default.removeItem(at: modelURL)
    }

    static func loadSettings() -> AppSettings {
        guard let data = try? Data(contentsOf: settingsURL),
              var settings = try? JSONDecoder().decode(AppSettings.self, from: data) else {
            return AppSettings()
        }
        // AeroMouse ships exactly two gestures. Settings from older builds may still have tap,
        // scroll, fist or double-click enabled, which is what made the pointer unpredictable;
        // force the simple gesture set on load instead of trusting old saved values.
        settings.gestures.tapEnabled = false
        settings.gestures.scrollEnabled = false
        settings.gestures.freezeEnabled = false
        settings.gestures.doubleClickEnabled = false
        settings.gestures.pinchClickEnabled = false
        settings.gestures.rightClickEnabled = false
        settings.gestures.dragEnabled = false
        settings.eyeAssistEnabled = false
        return settings
    }

    static func saveSettings(_ settings: AppSettings) {
        ensureDirectory()
        guard let data = try? JSONEncoder().encode(settings) else { return }
        try? data.write(to: settingsURL, options: .atomic)
    }
}
