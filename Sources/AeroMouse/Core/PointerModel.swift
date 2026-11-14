import Foundation

// MARK: - Trust gate

/// Prevents the neural corrector from extrapolating into territory it never saw. Far outside
/// the calibrated envelope the model fades out and the geometric mapping takes over.
struct TrustGate: Codable, Equatable {
    var points: [[Double]] = []
    var inner: Double = 1.0
    var outer: Double = 3.2

    func weight(for features: [Double]) -> Double {
        guard !points.isEmpty else { return 0 }
        var bestSquared = Double.infinity
        for point in points {
            var sum = 0.0
            let n = min(point.count, features.count)
            for i in 0..<n {
                let d = point[i] - features[i]
                sum += d * d
            }
            if sum < bestSquared { bestSquared = sum }
        }
        let distance = bestSquared.squareRoot()
        if !distance.isFinite { return 0 }
        if distance <= inner { return 1 }
        if distance >= outer { return 0 }
        let t = (distance - inner) / (outer - inner)
        let smooth = t * t * (3 - 2 * t)
        return 1 - smooth
    }
}

// MARK: - Gaze model (beta)

struct GazeModel: Codable, Equatable {
    var coeffsU: [Double]
    var coeffsV: [Double]
    var rmsPixels: Double

    static func features(_ gaze: GazeFeatures) -> [Double] {
        [1, gaze.irisX, gaze.irisY, gaze.yaw, gaze.roll]
    }

    func predict(_ gaze: GazeFeatures) -> Vec2 {
        let f = GazeModel.features(gaze)
        var u = 0.0
        var v = 0.0
        for i in 0..<min(f.count, coeffsU.count) { u += f[i] * coeffsU[i] }
        for i in 0..<min(f.count, coeffsV.count) { v += f[i] * coeffsV[i] }
        return Vec2(u, v)
    }

    /// Ridge regression from iris/head-pose features to screen position. Fitted implicitly
    /// while you look at the calibration targets.
    static func fit(_ pairs: [(features: GazeFeatures, target: Vec2)], screen: RectD) -> GazeModel? {
        guard pairs.count >= 30 else { return nil }
        let distinct = Set(pairs.map { "\(Int($0.target.x * 24)):\(Int($0.target.y * 24))" }).count
        guard distinct >= 5 else { return nil }

        let dim = 5
        var ata = [[Double]](repeating: [Double](repeating: 0, count: dim), count: dim)
        var bu = [Double](repeating: 0, count: dim)
        var bv = [Double](repeating: 0, count: dim)

        for pair in pairs {
            let f = GazeModel.features(pair.features)
            for a in 0..<dim {
                bu[a] += f[a] * pair.target.x
                bv[a] += f[a] * pair.target.y
                for b in 0..<dim { ata[a][b] += f[a] * f[b] }
            }
        }
        let trace = (0..<dim).reduce(0.0) { $0 + ata[$1][$1] }
        let lambda = max(trace * 0.002, 1e-9)
        for a in 0..<dim { ata[a][a] += lambda }

        guard let cu = Linalg.solve(ata, bu), let cv = Linalg.solve(ata, bv) else { return nil }

        var total = 0.0
        for pair in pairs {
            let prediction = Vec2(
                (0..<dim).reduce(0.0) { $0 + GazeModel.features(pair.features)[$1] * cu[$1] },
                (0..<dim).reduce(0.0) { $0 + GazeModel.features(pair.features)[$1] * cv[$1] }
            )
            let dx = (prediction.x - pair.target.x) * screen.width
            let dy = (prediction.y - pair.target.y) * screen.height
            total += dx * dx + dy * dy
        }
        let rms = (total / Double(pairs.count)).squareRoot()
        guard rms.isFinite else { return nil }
        return GazeModel(coeffsU: cu, coeffsV: cv, rmsPixels: rms)
    }
}

// MARK: - Calibration statistics

/// Per-target accuracy, used by the accuracy chart.
struct CellError: Codable, Equatable {
    var target: Vec2          // normalized screen position
    var count: Int
    var medianBefore: Double  // px
    var medianAfter: Double   // px
    var bias: Vec2            // mean signed error in px (after the full stack)
}

struct CalibrationStats: Codable {
    var samples: Int = 0
    var targets: Int = 0
    var rmsBefore: Double = 0
    var rmsAfter: Double = 0
    var medianBefore: Double = 0
    var medianAfter: Double = 0
    var p95Before: Double = 0
    var p95After: Double = 0
    var chosen: [String] = []
    var residualSplineIncluded: Bool = false
    var residualNetIncluded: Bool = false
    var residualNetValidationLoss: Double = 0
    var gazeRMS: Double?
    var screenWidth: Double = 0
    var screenHeight: Double = 0
    var trainedAt: Date = Date()

    // Detailed view, drawn as the accuracy chart.
    var cells: [CellError] = []
    var errorsBefore: [Double] = []   // px, one per held-out sample
    var errorsAfter: [Double] = []
}

// MARK: - Pointer model

/// The complete accuracy stack: a cross-validated blend of projective/quadratic/cubic/spline
/// fits, optionally corrected by a neural residual net conditioned on hand pose.
struct PointerModel: Codable {
    // 4: aspect-corrected coordinates and the middle-phalanx anchor.
    var version: Int = 4
    var screen: RectD
    var maps: [FittedMap]
    /// Second stage: a spline fitted on the residual of the blended maps, in output space.
    var residualStage: FittedMap?
    /// Third stage: local weighted-quadratic correction for curvature a global fit cannot follow.
    var localResidual: LocalResidual?
    var mlp: MLP?
    var scaler: StandardScaler?
    var trust: TrustGate?
    var referenceScale: Double = 0.15
    var gaze: GazeModel?
    var stats: CalibrationStats

    // MARK: Blending

    static func blend(_ maps: [(FittedMap, Double)], at point: Vec2, pose: PointerPose) -> Vec2? {
        var total = Vec2.zero
        var weightSum = 0.0
        for (map, weight) in maps {
            guard weight > 0, let mapped = map.raw(point, pose: pose), mapped.isValid else { continue }
            total += mapped * weight
            weightSum += weight
        }
        guard weightSum > 0 else { return nil }
        return total / weightSum
    }

    func ensemble(_ point: Vec2, pose: PointerPose) -> Vec2 {
        PointerModel.blend(maps.map { ($0, $0.weight) }, at: point, pose: pose) ?? point
    }

    // MARK: Mapping

    /// Returns screen-normalized (0...1, origin top-left) for a raw image-space anchor.
    func normalized(for source: Vec2, pose: PointerPose) -> Vec2 {
        guard source.isValid else { return Vec2(0.5, 0.5) }
        var point = ensemble(source, pose: pose)

        // Cascade, not average: the second stage only has to learn what the first missed.
        if let residualStage, let corrected = residualStage.raw(point, pose: pose), corrected.isValid {
            point = corrected
        }

        // Then the local stage, which knows where the calibration actually bends. It returns
        // nothing outside the sampled area, so the global fit keeps charge of the edges.
        if let localResidual, let correction = localResidual.residual(at: point) {
            point += correction
        }

        if let mlp, let scaler, let trust {
            let features = PointerFeatures.vector(mapped: point, pose: pose, referenceScale: referenceScale)
            let scaled = scaler.transform(features)
            let confidence = trust.weight(for: scaled)
            if confidence > 0 {
                let residual = mlp.predict(scaled)
                if residual.count >= 2, residual[0].isFinite, residual[1].isFinite {
                    // Never let the learned correction dominate the geometric mapping.
                    let limit = 0.035
                    let rx = min(max(residual[0] * confidence, -limit), limit)
                    let ry = min(max(residual[1] * confidence, -limit), limit)
                    point += Vec2(rx, ry)
                }
            }
        }

        return Vec2(min(max(point.x, -0.02), 1.02), min(max(point.y, -0.02), 1.02))
    }

    func screenPoint(for source: Vec2, pose: PointerPose) -> Vec2 {
        screen.denormalized(normalized(for: source, pose: pose))
    }

    var isCalibrated: Bool { !maps.isEmpty && stats.samples > 0 }
}
