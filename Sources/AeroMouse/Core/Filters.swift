import Foundation

// MARK: - Settings surface for the pointer pipeline

struct SmoothingConfig: Codable {
    var minCutoff: Double = 1.6          // Hz. Lower = silkier, higher = snappier.
    var beta: Double = 0.028             // Speed coupling: lets fast motion through almost unfiltered.
    var derivativeCutoff: Double = 1.0   // Smooths the speed estimate, which keeps beta from firing on noise.
    var predictionMs: Double = 16        // Constant-velocity lookahead to cancel filter lag.
    var predictionSpeedLow: Double = 80  // px/s below which prediction is faded out
    var predictionSpeedHigh: Double = 360
    var precisionGain: Double = 0.65     // Extra gain reduction while moving slowly.
    var precisionSpeedLow: Double = 50   // px/s where precision mode starts
    var precisionSpeedHigh: Double = 320 // px/s where full gain resumes
    var deadband: Double = 0.9           // px of hysteresis before the cursor is re-emitted
    var outlierGate: Double = 180        // px jump rejected outright
}

// MARK: - One Euro filter

/// The One Euro filter: an adaptive low-pass whose cutoff rises with speed. It is the standard
/// remedy for the "smooth but laggy / responsive but shaky" trade-off in pointer tracking.
final class OneEuroFilter2D {
    private var previous: Vec2?
    private var derivative = Vec2.zero
    private var previousTime: Double?

    /// Low-passed pointer speed in px/s. Prediction reuses this instead of a raw difference,
    /// which otherwise turns sensor noise into visible overshoot.
    var smoothedDerivative: Vec2 { derivative }

    func reset(_ point: Vec2? = nil, time: Double? = nil) {
        previous = point
        derivative = .zero
        previousTime = time
    }

    func filter(_ value: Vec2, time: Double, config: SmoothingConfig) -> Vec2 {
        guard let lastPoint = previous, let lastTime = previousTime else {
            previous = value
            previousTime = time
            return value
        }
        let dt = max(min(time - lastTime, 0.2), 1.0 / 1000.0)
        let rawDerivative = (value - lastPoint) / dt
        let alphaD = Self.alpha(cutoff: config.derivativeCutoff, dt: dt)
        derivative = derivative + (rawDerivative - derivative) * alphaD

        let cutoff = max(config.minCutoff + config.beta * derivative.length, 0.05)
        let alpha = Self.alpha(cutoff: cutoff, dt: dt)
        let result = lastPoint + (value - lastPoint) * alpha

        previous = result
        previousTime = time
        return result
    }

    private static func alpha(cutoff: Double, dt: Double) -> Double {
        let tau = 1.0 / (2.0 * Double.pi * max(cutoff, 0.001))
        return 1.0 / (1.0 + tau / dt)
    }
}

// MARK: - Click lock

/// A pinch physically drags the fingertip a couple of centimetres. Without compensation the
/// cursor twitches the moment you click. This absorbs that motion and decays it away, so the
/// click lands where you aimed and the pointer glides back underneath your finger.
final class ClickLock {
    private var offset = Vec2.zero
    private let timeConstant = 0.20

    var isActive: Bool { offset.length > 0.05 }

    func begin(offset: Vec2) {
        self.offset = offset
    }

    func decay(dt: Double) {
        guard offset.length > 0 else { return }
        offset = offset * exp(-max(dt, 0) / timeConstant)
        if offset.length < 0.05 { offset = .zero }
    }

    func apply(_ point: Vec2) -> Vec2 { point - offset }

    func reset() { offset = .zero }
}

// MARK: - Pointer pipeline

struct PointerPipelineOutput {
    var point: Vec2
    var velocity: Vec2
    var raw: Vec2
    var held: Bool
}

/// Turns a jittery per-frame model point into a stable integer cursor position.
final class PointerPipeline {
    private var filter = OneEuroFilter2D()
    private var lastModel: Vec2?
    private var lastVirtual: Vec2?
    private var lastTime: Double?
    private var velocity = Vec2.zero
    private var emitted: Vec2?
    private var rejectCount = 0

    func reset(at point: Vec2?) {
        filter.reset(point)
        lastModel = point
        lastVirtual = point
        emitted = point
        velocity = .zero
        lastTime = nil
        rejectCount = 0
    }

    func process(modelPoint: Vec2?, time: Double, bounds: RectD, config: SmoothingConfig) -> PointerPipelineOutput? {
        guard let raw = modelPoint, raw.isValid, bounds.isValid else { return nil }
        let clamped = raw.clamped(to: bounds)

        guard let virtualBase = lastVirtual, let previousTime = lastTime else {
            filter.reset(clamped, time: time)
            lastVirtual = clamped
            lastModel = clamped
            lastTime = time
            emitted = clamped
            return PointerPipelineOutput(point: clamped, velocity: .zero, raw: clamped, held: false)
        }

        let dt = max(min(time - previousTime, 0.2), 1.0 / 1000.0)

        // Reject single-frame teleports (a landmark popping onto the wrong finger, a dropped
        // frame): hold the cursor instead of throwing it across the screen.
        let predicted = virtualBase + velocity * dt
        if clamped.distance(to: predicted) > config.outlierGate {
            rejectCount += 1
            if rejectCount < 4 {
                return PointerPipelineOutput(point: emitted ?? clamped, velocity: velocity, raw: clamped, held: true)
            }
            filter.reset(clamped, time: time)
            lastVirtual = clamped
            lastModel = clamped
            lastTime = time
            emitted = nil
            velocity = .zero
            rejectCount = 0
            return PointerPipelineOutput(point: clamped, velocity: .zero, raw: clamped, held: false)
        }
        rejectCount = 0

        // Precision mode: slow motion is attenuated so small targets are reachable.
        let rawSpeed = lastModel.map { clamped.distance(to: $0) / dt } ?? 0
        let gain = Self.precisionGain(speed: rawSpeed, config: config)
        let delta = (clamped - virtualBase) * gain
        let virtual = virtualBase + delta

        let filtered = filter.filter(virtual, time: time, config: config)
        velocity = filter.smoothedDerivative

        lastVirtual = virtual
        lastModel = clamped
        lastTime = time

        // Prediction only while genuinely moving. At rest it would amplify whatever residual
        // noise the filter leaves behind, which is exactly the shimmer we are trying to kill.
        let predictionWeight = Self.smoothstep(velocity.length, config.predictionSpeedLow, config.predictionSpeedHigh)
        var target = filtered + velocity * (config.predictionMs / 1000.0 * predictionWeight)

        // Deadband: while settling, keep the exact previous position so the pointer cannot
        // shimmer by a pixel. This is the difference between "smooth" and "butter".
        let deadband = config.deadband * (velocity.length < 25 ? 1.35 : 1.0)
        if let previousEmission = emitted,
           target.distance(to: previousEmission) < deadband,
           velocity.length < 45 {
            target = previousEmission
        }

        let rounded = Vec2(target.x.rounded(), target.y.rounded()).clamped(to: bounds)
        emitted = rounded
        return PointerPipelineOutput(point: rounded, velocity: velocity, raw: clamped, held: false)
    }

    static func smoothstep(_ value: Double, _ low: Double, _ high: Double) -> Double {
        let lo = min(low, high)
        let hi = max(low, high)
        if value <= lo { return 0 }
        if value >= hi { return 1 }
        let t = (value - lo) / (hi - lo)
        return t * t * (3 - 2 * t)
    }

    private static func precisionGain(speed: Double, config: SmoothingConfig) -> Double {
        let low = max(config.precisionSpeedLow, 1)
        let high = max(config.precisionSpeedHigh, low + 1)
        if speed <= low { return config.precisionGain }
        if speed >= high { return 1 }
        let t = (speed - low) / (high - low)
        let smooth = t * t * (3 - 2 * t)   // smoothstep, no gain step discontinuities
        return config.precisionGain + (1 - config.precisionGain) * smooth
    }

    /// Exposed for the self-test so the gain curve can be asserted.
    static func selfTestsGainAtSpeed(_ speed: Double) -> Double {
        precisionGain(speed: speed, config: SmoothingConfig())
    }
}

// MARK: - Jitter measurement (used for the live accuracy readout)

final class JitterMeter {
    private var last: Vec2?
    private var lastTime: Double?
    private var samples: [Double] = []

    func record(_ point: Vec2, time: Double) {
        defer {
            last = point
            lastTime = time
        }
        guard let previous = last, let previousTime = lastTime else { return }
        let dt = time - previousTime
        guard dt > 0 else { return }
        let speed = point.distance(to: previous) / dt
        guard speed < 40 else {   // only measure while effectively still
            samples.removeAll(keepingCapacity: true)
            return
        }
        samples.append(point.distance(to: previous))
        if samples.count > 45 { samples.removeFirst() }
    }

    /// RMS of per-frame movement while still, in pixels. Lower is steadier.
    var jitterRMS: Double {
        guard samples.count >= 4 else { return 0 }
        let mean = samples.reduce(0, +) / Double(samples.count)
        let variance = samples.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / Double(samples.count)
        return variance.squareRoot()
    }

    func reset() {
        last = nil
        lastTime = nil
        samples.removeAll(keepingCapacity: true)
    }
}
