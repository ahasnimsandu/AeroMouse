import Foundation

/// Detects a deliberate both-eyes blink from the eye-opening signal.
///
/// A natural blink is short (~0.1 s) and must not click, so a blink only counts after the eyes
/// have stayed closed for a minimum duration. The open baseline adapts to the person, camera and
/// lighting, because the absolute eye-aspect ratio differs wildly between faces.
struct BlinkDetector {
    var openBaseline: Double = 0.35
    var minClosedDuration: Double = 0.20
    var maxClosedDuration: Double = 0.90
    var cooldown: Double = 0.35
    var closedThresholdFloor: Double = 0.12
    var openThresholdFloor: Double = 0.20

    private(set) var closedSince: Double = 0
    private(set) var lastBlink: Double = -10

    var closedThreshold: Double { max(closedThresholdFloor, openBaseline * 0.45) }
    var openThreshold: Double { max(openThresholdFloor, openBaseline * 0.70) }

    mutating func update(openness: Double, time: Double) -> Bool {
        if openness > 0.15 {
            openBaseline = openBaseline * 0.95 + openness * 0.05
        }
        let closed = openness < closedThreshold
        let open = openness > openThreshold

        if closed {
            if closedSince == 0 { closedSince = time }
            return false
        }
        guard open, closedSince > 0 else { return false }
        let duration = time - closedSince
        closedSince = 0
        guard duration >= minClosedDuration, duration <= maxClosedDuration,
              time - lastBlink > cooldown else { return false }
        lastBlink = time
        return true
    }

    mutating func reset() {
        closedSince = 0
    }
}
