import Foundation

/// Hand pose descriptors that a homography cannot capture (depth, wrist roll, finger spread).
/// Feeding these to the residual network is what makes the mapping depth- and posture-aware.
struct PointerPose: Codable, Equatable {
    var handScale: Double = 0.15
    var roll: Double = 0
    var spread: Double = 1
}

/// Everything a webcam can tell you about where someone is looking: both pupils inside their eye
/// openings, head orientation, head position and how open the eyes are.
struct GazeFeatures: Codable, Equatable {
    var irisX: Double = 0        // mean pupil offset, −1…1 across the eye opening
    var irisY: Double = 0
    var yaw: Double = 0          // radians
    var roll: Double = 0
    var leftX: Double = 0
    var leftY: Double = 0
    var rightX: Double = 0
    var rightY: Double = 0
    var pitch: Double = 0
    var headX: Double = 0.5      // face box centre, image normalized (bottom-left origin)
    var headY: Double = 0.5
    var headScale: Double = 0.25 // face width in aspect-corrected units: distance from the camera
    var openness: Double = 1     // 1 = open, 0 = blinking

    /// Feature vector for the gaze regression. The quadratic terms let the projection bend, which
    /// a plain linear fit cannot follow, and the head terms let the model tell "eyes moved" from
    /// "whole head moved" — the difference between accurate and useless on a camera with no depth.
    var vector: [Double] {
        let logDistance = log(max(headScale, 1e-4) / 0.25)
        return [
            1,
            leftX, leftY, rightX, rightY,
            yaw, pitch, roll,
            headX - 0.5, headY - 0.5, logDistance,
            leftX * leftY, rightX * rightY,
            yaw * pitch, yaw * roll,
        ]
    }

    static let dimension = 15
}

/// One calibration observation: where the tracker said the hand was, and where the screen
/// point actually was. `group` ties repeated passes of the same physical target together so
/// cross-validation never scores a model on a location it effectively memorised.
struct CalibrationSample: Codable {
    var source: Vec2          // anchor, normalized image coordinates (Vision, origin bottom-left)
    var target: Vec2          // screen normalized, origin top-left
    var weight: Double        // pinch samples weigh more than passive hover samples
    var group: Int
    var pose: PointerPose
    var gaze: GazeFeatures?
}

enum PointerFeatures {
    static let dimension = 5

    /// Inputs for the neural residual corrector.
    static func vector(mapped: Vec2, pose: PointerPose, referenceScale: Double) -> [Double] {
        let ref = max(referenceScale, 1e-4)
        let scale = max(pose.handScale, 1e-4)
        return [mapped.x, mapped.y, log(scale / ref), pose.roll, pose.spread]
    }
}

/// Geometric scale of an error distribution, in pixels.
struct ErrorMetrics {
    var rms: Double = .infinity
    var median: Double = .infinity
    var p95: Double = .infinity
    var count: Int = 0

    static func compute(predicted: [Vec2], targets: [Vec2], weights: [Double], screen: RectD) -> ErrorMetrics {
        guard !predicted.isEmpty, predicted.count == targets.count else { return ErrorMetrics() }
        var rows: [(d: Double, w: Double)] = []
        rows.reserveCapacity(predicted.count)
        for i in 0..<predicted.count {
            let dx = (predicted[i].x - targets[i].x) * screen.width
            let dy = (predicted[i].y - targets[i].y) * screen.height
            let d = (dx * dx + dy * dy).squareRoot()
            guard d.isFinite else { continue }
            rows.append((d, max(weights.indices.contains(i) ? weights[i] : 1, 1e-6)))
        }
        guard !rows.isEmpty else { return ErrorMetrics() }
        rows.sort { $0.d < $1.d }
        let totalW = rows.reduce(0.0) { $0 + $1.w }
        guard totalW > 0 else { return ErrorMetrics() }
        let rms = (rows.reduce(0.0) { $0 + $1.w * $1.d * $1.d } / totalW).squareRoot()
        let median = weightedQuantile(rows, 0.50, totalW)
        let p95 = weightedQuantile(rows, 0.95, totalW)
        return ErrorMetrics(rms: rms, median: median, p95: p95, count: rows.count)
    }

    private static func weightedQuantile(_ rows: [(d: Double, w: Double)], _ q: Double, _ totalW: Double) -> Double {
        let goal = q * totalW
        var acc = 0.0
        for row in rows {
            acc += row.w
            if acc >= goal { return row.d }
        }
        return rows.last?.d ?? .infinity
    }
}
