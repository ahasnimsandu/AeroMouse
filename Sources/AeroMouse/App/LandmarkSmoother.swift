import Foundation
import Vision

/// Per-landmark temporal filtering with confidence gating.
///
/// The anchor used to be the only filtered quantity, but in poor light the derived measurements
/// (hand scale, wrist roll, thumb-index distance) are computed from noisy joints — and those feed
/// the pinch detector and the pose features of the model. Filtering all 21 joints, damping
/// implausible jumps on weak detections, and briefly holding dropped joints stabilises all of it.
final class LandmarkSmoother {
    private var filters: [VNHumanHandPoseObservation.JointName: OneEuroFilter2D] = [:]
    private var lastPoints: [VNHumanHandPoseObservation.JointName: Vec2] = [:]
    private var lastSeen: [VNHumanHandPoseObservation.JointName: Double] = [:]
    private var lastConfidence: [VNHumanHandPoseObservation.JointName: Double] = [:]

    /// Joints move much faster than the pointer, so this filter is deliberately snappier than the
    /// cursor filter: its job is to remove sensor noise, not to smooth intentional motion.
    private var config: SmoothingConfig = {
        var config = SmoothingConfig()
        config.minCutoff = 5.0
        config.beta = 0.08
        config.derivativeCutoff = 2.0
        config.predictionMs = 0
        config.deadband = 0
        config.precisionGain = 1
        config.outlierGate = 1e9
        return config
    }()

    private let holdSeconds: Double = 0.22

    func smooth(_ points: [VNHumanHandPoseObservation.JointName: Vec2],
                confidences: [VNHumanHandPoseObservation.JointName: Double],
                time: Double) -> [VNHumanHandPoseObservation.JointName: Vec2] {
        var output: [VNHumanHandPoseObservation.JointName: Vec2] = [:]

        for (name, point) in points {
            let confidence = confidences[name] ?? 0
            // Below this the network is guessing. Holding the last good position for a moment is
            // far safer than following a joint that may be on the wrong finger.
            guard confidence > 0.30 else { continue }

            let filter: OneEuroFilter2D
            if let existing = filters[name] {
                filter = existing
            } else {
                filter = OneEuroFilter2D()
                filters[name] = filter
            }

            var target = point
            if let previous = lastPoints[name] {
                let previousConfidence = lastConfidence[name] ?? 1
                // A weak detection that jumps implausibly far is usually the wrong finger or a
                // smear in the dark: follow it only partially instead of snapping to it.
                if confidence < 0.5 || previousConfidence < 0.35 {
                    let jump = point.distance(to: previous)
                    if jump > 0.05 {
                        target = Vec2.lerp(previous, point, 0.4)
                    }
                }
            }

            let smoothed = filter.filter(target, time: time, config: config)
            output[name] = smoothed
            lastPoints[name] = smoothed
            lastSeen[name] = time
            lastConfidence[name] = confidence
        }

        // Brief dropouts: keep the last position so the skeleton does not flicker, but drop stale
        // joints so a hand that has left the frame cannot keep producing phantom pinch distances.
        for (name, previous) in lastPoints where output[name] == nil {
            if let seen = lastSeen[name], time - seen < holdSeconds {
                output[name] = previous
            } else {
                filters[name] = nil
                lastPoints[name] = nil
                lastSeen[name] = nil
                lastConfidence[name] = nil
            }
        }

        return output
    }

    func reset() {
        filters.removeAll()
        lastPoints.removeAll()
        lastSeen.removeAll()
        lastConfidence.removeAll()
    }
}
