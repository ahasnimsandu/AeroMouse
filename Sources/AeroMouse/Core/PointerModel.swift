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

/// Maps webcam gaze to a screen position.
///
/// Both pupils, head orientation, head position and eye opening all take part, with quadratic
/// terms, because the projection of a pupil onto a screen is not linear and a head that turns moves
/// the eyes in the face without changing where they point. The reported accuracy is measured by
/// grouped cross-validation over the calibration targets, so it is an honest figure rather than a
/// fit residual — and a model that cannot beat a useful threshold is discarded instead of being
/// applied weakly.
struct GazeModel: Codable, Equatable {
    var coeffsU: [Double]
    var coeffsV: [Double]
    var rmsPixels: Double
    var cvPixels: Double = 0
    var samples: Int = 0

    /// Nil when the model's feature layout does not match this build (an older calibration) or
    /// when the prediction leaves the screen by a wide margin.
    func predict(_ gaze: GazeFeatures) -> Vec2? {
        let f = gaze.vector
        guard f.count == coeffsU.count, coeffsU.count == coeffsV.count else { return nil }
        var u = 0.0
        var v = 0.0
        for i in 0..<f.count {
            u += f[i] * coeffsU[i]
            v += f[i] * coeffsV[i]
        }
        let point = Vec2(u, v)
        guard point.isValid, abs(point.x - 0.5) < 0.8, abs(point.y - 0.5) < 0.8 else { return nil }
        return point
    }

    typealias Pair = (features: GazeFeatures, target: Vec2, weight: Double)

    private static func solve(_ pairs: [Pair], _ coordinate: (Vec2) -> Double) -> [Double]? {
        let dimension = GazeFeatures.dimension
        var rows: [[Double]] = []
        var rhs: [Double] = []
        rows.reserveCapacity(pairs.count + dimension)

        for pair in pairs {
            let root = max(pair.weight, 1e-3).squareRoot()
            rows.append(pair.features.vector.map { $0 * root })
            rhs.append(coordinate(pair.target) * root)
        }
        guard rows.count > dimension else { return nil }

        var energy = 0.0
        for row in rows { for value in row { energy += value * value } }
        let damping = max(energy / Double(rows.count * dimension) * 1e-4, 1e-14).squareRoot()
        for column in 0..<dimension {
            var row = [Double](repeating: 0, count: dimension)
            row[column] = damping
            rows.append(row)
            rhs.append(0)
        }
        return Linalg.leastSquares(rows, rhs)
    }

    private static func pixelRMS(_ coeffsU: [Double], _ coeffsV: [Double],
                                 _ pairs: [Pair], screen: RectD) -> Double {
        guard !pairs.isEmpty else { return .infinity }
        var total = 0.0
        var weightSum = 0.0
        for pair in pairs {
            let f = pair.features.vector
            guard f.count == coeffsU.count, f.count == coeffsV.count else { return .infinity }
            var u = 0.0
            var v = 0.0
            for i in 0..<f.count {
                u += f[i] * coeffsU[i]
                v += f[i] * coeffsV[i]
            }
            let dx = (u - pair.target.x) * screen.width
            let dy = (v - pair.target.y) * screen.height
            guard dx.isFinite, dy.isFinite else { continue }
            let w = max(pair.weight, 1e-3)
            total += w * (dx * dx + dy * dy)
            weightSum += w
        }
        guard weightSum > 0 else { return .infinity }
        return (total / weightSum).squareRoot()
    }

    /// Fitted implicitly while you look at the calibration targets.
    static func fit(_ pairs: [Pair], screen: RectD) -> GazeModel? {
        // Blink frames and closed eyes carry no gaze at all; including them poisons the fit.
        let usable = pairs.filter { $0.features.openness > 0.25 && $0.target.isValid
                                    && $0.target.x.isFinite && $0.target.y.isFinite }
        guard usable.count >= 40 else { return nil }
        let cells = Set(usable.map { "\(Int($0.target.x * 20)):\(Int($0.target.y * 20))" })
        guard cells.count >= 6 else { return nil }

        guard let cu = solve(usable, { $0.x }), let cv = solve(usable, { $0.y }) else { return nil }

        // Grouped k-fold over target cells: the error you would actually get at a spot you did not
        // calibrate on, not the residual of the fit itself.
        var foldOf: [String: Int] = [:]
        for (index, cell) in cells.sorted().enumerated() { foldOf[cell] = index % 4 }

        var total = 0.0
        var weightSum = 0.0
        for fold in 0..<4 {
            let training = usable.filter { foldOf["\(Int($0.target.x * 20)):\(Int($0.target.y * 20))"] != fold }
            let testing = usable.filter { foldOf["\(Int($0.target.x * 20)):\(Int($0.target.y * 20))"] == fold }
            guard training.count > GazeFeatures.dimension, !testing.isEmpty else { continue }
            guard let fu = solve(training, { $0.x }), let fv = solve(training, { $0.y }) else { continue }
            for pair in testing {
                let f = pair.features.vector
                guard f.count == fu.count, f.count == fv.count else { continue }
                var u = 0.0
                var v = 0.0
                for i in 0..<f.count {
                    u += f[i] * fu[i]
                    v += f[i] * fv[i]
                }
                let dx = (u - pair.target.x) * screen.width
                let dy = (v - pair.target.y) * screen.height
                guard dx.isFinite, dy.isFinite else { continue }
                let w = max(pair.weight, 1e-3)
                total += w * (dx * dx + dy * dy)
                weightSum += w
            }
        }

        let cvRMS = weightSum > 0 ? (total / weightSum).squareRoot() : .infinity
        let inSample = pixelRMS(cu, cv, usable, screen: screen)
        let reported = cvRMS.isFinite ? cvRMS : inSample
        guard reported.isFinite else { return nil }
        // A gaze model that cannot land within this is not helping anything; refusing it is
        // better than pulling the pointer toward the wrong place.
        guard reported < 170 else { return nil }

        return GazeModel(coeffsU: cu, coeffsV: cv, rmsPixels: reported, cvPixels: cvRMS.isFinite ? cvRMS : 0,
                         samples: usable.count)
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
