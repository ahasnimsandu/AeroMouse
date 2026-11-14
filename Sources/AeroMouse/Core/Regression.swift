import Foundation

// MARK: - Shared ridge solver

enum RidgeSolver {
    /// Weighted ridge regression solved by QR. The damping rows are appended as `sqrt(lambda) * I`,
    /// so the result is a true regularised least-squares solution rather than a squared-condition
    /// normal-equation guess.
    static func fit(samples: [CalibrationSample],
                    dimension: Int,
                    ridge: Double = 1e-9,
                    basis: (CalibrationSample) -> [Double]) -> (x: [Double], y: [Double])? {
        guard samples.count >= 4, dimension >= 1 else { return nil }

        var rows: [[Double]] = []
        var bx: [Double] = []
        var by: [Double] = []
        rows.reserveCapacity(samples.count + dimension)

        for sample in samples {
            let f = basis(sample)
            guard f.count == dimension else { continue }
            let rootWeight = max(sample.weight, 1e-6).squareRoot()
            rows.append(f.map { $0 * rootWeight })
            bx.append(sample.target.x * rootWeight)
            by.append(sample.target.y * rootWeight)
        }
        guard rows.count >= 3 else { return nil }

        var energy = [Double](repeating: 0, count: dimension)
        for row in rows {
            for j in 0..<dimension { energy[j] += row[j] * row[j] }
        }
        let meanEnergy = energy.reduce(0, +) / Double(dimension)
        let damping = max(ridge * meanEnergy, 1e-16).squareRoot()

        for j in 0..<dimension {
            var row = [Double](repeating: 0, count: dimension)
            row[j] = damping
            rows.append(row)
            bx.append(0)
            by.append(0)
        }

        guard let x = Linalg.leastSquares(rows, bx),
              let y = Linalg.leastSquares(rows, by) else { return nil }
        return (x, y)
    }
}

// MARK: - Polynomial regression (ridge least squares on a total-degree basis)

struct PolynomialMap: Codable, Equatable {
    var degree: Int
    var coeffsX: [Double]
    var coeffsY: [Double]

    /// Centered on the middle of the area. Without this the cubic basis conditions like a
    /// Vandermonde matrix and the fit degrades badly.
    static func basis(_ p: Vec2, degree: Int) -> [Double] {
        let d = max(degree, 0)
        let x = p.x - 0.5
        let y = p.y - 0.5
        var out: [Double] = []
        out.reserveCapacity((d + 1) * (d + 2) / 2)
        for total in 0...d {
            for i in stride(from: total, through: 0, by: -1) {
                let j = total - i
                out.append(pow(x, Double(i)) * pow(y, Double(j)))
            }
        }
        return out
    }

    static func termCount(degree: Int) -> Int {
        let d = max(degree, 0)
        return (d + 1) * (d + 2) / 2
    }

    func map(_ p: Vec2) -> Vec2 {
        let f = PolynomialMap.basis(p, degree: degree)
        var x = 0.0
        var y = 0.0
        for i in 0..<min(f.count, coeffsX.count) { x += f[i] * coeffsX[i] }
        for i in 0..<min(f.count, coeffsY.count) { y += f[i] * coeffsY[i] }
        return Vec2(x, y)
    }

    static func fit(_ samples: [CalibrationSample], degree: Int, ridge: Double = 1e-9) -> PolynomialMap? {
        let dim = termCount(degree: degree)
        guard samples.count >= dim + 2 else { return nil }
        guard let solve = RidgeSolver.fit(samples: samples, dimension: dim, ridge: ridge, basis: {
            basis($0.source, degree: degree)
        }) else { return nil }
        return PolynomialMap(degree: degree, coeffsX: solve.x, coeffsY: solve.y)
    }
}

// MARK: - Pose-conditioned linear regression

/// Linear regression on interaction terms between screen position and hand pose. The `u * depth`
/// and `v * depth` terms are exactly the multiplicative stretch that happens when the hand moves
/// towards or away from the camera — something no pure homography can represent.
struct PoseRegressionMap: Codable, Equatable {
    var referenceScale: Double
    var coeffsX: [Double]
    var coeffsY: [Double]

    static let termCount = 12

    static func basis(_ source: Vec2, pose: PointerPose, referenceScale: Double) -> [Double] {
        let ref = max(referenceScale, 1e-4)
        let depth = log(max(pose.handScale, 1e-4) / ref)
        let roll = pose.roll
        let spread = pose.spread
        let x = source.x - 0.5
        let y = source.y - 0.5
        return [
            1, x, y,
            depth, roll, spread,
            x * depth, y * depth,
            x * roll, y * roll,
            x * spread, y * spread,
        ]
    }

    func map(_ source: Vec2, pose: PointerPose) -> Vec2 {
        let f = PoseRegressionMap.basis(source, pose: pose, referenceScale: referenceScale)
        var x = 0.0
        var y = 0.0
        for i in 0..<min(f.count, coeffsX.count) { x += f[i] * coeffsX[i] }
        for i in 0..<min(f.count, coeffsY.count) { y += f[i] * coeffsY[i] }
        return Vec2(x, y)
    }

    static func fit(_ samples: [CalibrationSample], referenceScale: Double) -> PoseRegressionMap? {
        guard let solve = RidgeSolver.fit(samples: samples, dimension: termCount, ridge: 1e-9, basis: {
            basis($0.source, pose: $0.pose, referenceScale: referenceScale)
        }) else { return nil }
        return PoseRegressionMap(referenceScale: referenceScale, coeffsX: solve.x, coeffsY: solve.y)
    }
}

// MARK: - Fitted map bundle entries

struct FittedMap: Codable, Equatable {
    enum Kind: String, Codable, CaseIterable {
        case homography
        case poly2
        case poly3
        case poseLinear

        var displayName: String {
            switch self {
            case .homography: return "Projective (homography)"
            case .poly2: return "Quadratic regression"
            case .poly3: return "Cubic regression"
            case .poseLinear: return "Pose-conditioned regression"
            }
        }

        var shortName: String {
            switch self {
            case .homography: return "homography"
            case .poly2: return "poly2"
            case .poly3: return "poly3"
            case .poseLinear: return "pose"
            }
        }
    }

    var kind: Kind
    var weight: Double
    /// Grouped cross-validation RMS in pixels, or 0 when the map was fitted outside cross-validation
    /// (a residual stage, for example).
    var cvRMS: Double
    var homography: Mat3?
    var polynomial: PolynomialMap?
    var poseRegression: PoseRegressionMap?

    func raw(_ p: Vec2, pose: PointerPose) -> Vec2? {
        switch kind {
        case .homography:
            return homography?.map(p)
        case .poly2, .poly3:
            return polynomial?.map(p)
        case .poseLinear:
            return poseRegression?.map(p, pose: pose)
        }
    }
}

enum MapFitter {
    static func fit(kind: FittedMap.Kind,
                    samples: [CalibrationSample],
                    maxCenters: Int = 220,
                    referenceScale: Double = 0.15) -> FittedMap? {
        guard samples.count >= 4 else { return nil }
        switch kind {
        case .homography:
            let src = samples.map { $0.source }
            let dst = samples.map { $0.target }
            guard let h = Mat3.fit(from: src, to: dst) else { return nil }
            return FittedMap(kind: .homography, weight: 1, cvRMS: 0,
                             homography: h, polynomial: nil, poseRegression: nil)
        case .poly2:
            guard let p = PolynomialMap.fit(samples, degree: 2) else { return nil }
            return FittedMap(kind: .poly2, weight: 1, cvRMS: 0,
                             homography: nil, polynomial: p, poseRegression: nil)
        case .poly3:
            guard let p = PolynomialMap.fit(samples, degree: 3) else { return nil }
            return FittedMap(kind: .poly3, weight: 1, cvRMS: 0,
                             homography: nil, polynomial: p, poseRegression: nil)
        case .poseLinear:
            guard let p = PoseRegressionMap.fit(samples, referenceScale: referenceScale) else { return nil }
            return FittedMap(kind: .poseLinear, weight: 1, cvRMS: 0,
                             homography: nil, polynomial: nil, poseRegression: p)
        }
    }
}

// MARK: - Grouped cross-validation

struct PredictionError {
    var predicted: Vec2
    var target: Vec2
    var weight: Double
    var group: Int
}

enum CalibrationCV {
    /// Runs grouped k-fold CV. Whole target locations are held out together, so a model cannot
    /// score well by memorising a neighbour that is 2 cm away.
    static func run(samples: [CalibrationSample],
                    folds requestedFolds: Int,
                    builder: ([CalibrationSample]) -> (CalibrationSample) -> Vec2) -> [PredictionError] {
        let groups = Set(samples.map { $0.group }).sorted()
        guard groups.count >= 2 else { return [] }
        let folds = max(2, min(requestedFolds, groups.count))

        var foldOf: [Int: Int] = [:]
        for (i, g) in groups.enumerated() { foldOf[g] = i % folds }

        var out: [PredictionError] = []
        for fold in 0..<folds {
            let training = samples.filter { foldOf[$0.group] != fold }
            let testing = samples.filter { foldOf[$0.group] == fold }
            guard !training.isEmpty, !testing.isEmpty else { continue }
            let predict = builder(training)
            for s in testing {
                let p = predict(s)
                guard p.isValid else { continue }
                out.append(PredictionError(predicted: p, target: s.target,
                                           weight: s.weight, group: s.group))
            }
        }
        return out
    }

    static func metrics(_ predictions: [PredictionError], screen: RectD) -> ErrorMetrics {
        guard !predictions.isEmpty else { return ErrorMetrics() }
        return ErrorMetrics.compute(predicted: predictions.map { $0.predicted },
                                    targets: predictions.map { $0.target },
                                    weights: predictions.map { $0.weight },
                                    screen: screen)
    }
}
