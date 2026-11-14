import Foundation

/// Local residual correction.
///
/// Every global stage in the stack (homography, polynomials, pose regression) is one smooth
/// function over the whole plane. The real hand→screen map does not behave that way: near the
/// corners of the calibrated area it bends, and a single quadratic has to average that bend away.
/// Calibration errors between the dots — where nobody measures — are exactly where a global fit
/// hurts and where a local one wins.
///
/// For each query point this re-fits a tiny weighted quadratic using only the nearest calibration
/// samples, centred on the query so the fit is well conditioned. Where samples are sparse the
/// effective weight collapses and the correction fades out rather than extrapolating.
struct LocalResidual: Codable, Equatable {
    /// Flat table of mapped position and residual: x, y, rx, ry per entry.
    var samples: [Double] = []
    var bandwidth: Double = 0.16
    var neighbors: Int = 40
    var minSupport: Double = 8
    var ridge: Double = 1e-3
    /// 1 = local linear (steady), 2 = local quadratic (follows tighter curvature, less stable).
    var order: Int = 1

    var count: Int { samples.count / 4 }

    /// Kernel width, order and neighbourhood were picked by sweeping them against a synthetic
    /// truth that bends between the calibration dots (see the self-test): a narrow kernel has too
    /// little support and a wide one flattens the curvature again.
    init?(predictions: [(mapped: Vec2, target: Vec2)],
          bandwidth: Double = 0.16,
          order: Int = 2,
          neighbors: Int = 40) {
        self.bandwidth = bandwidth
        self.order = order
        self.neighbors = neighbors
        samples.reserveCapacity(predictions.count * 4)
        for prediction in predictions where prediction.mapped.isValid && prediction.target.isValid {
            let mapped = prediction.mapped
            samples.append(mapped.x)
            samples.append(mapped.y)
            samples.append(prediction.target.x - mapped.x)
            samples.append(prediction.target.y - mapped.y)
        }
        guard count >= 12 else { return nil }
    }

    /// Residual to add at `point`, or nil where the neighbourhood is too thin to trust.
    func residual(at point: Vec2) -> Vec2? {
        guard point.isValid, count >= 6 else { return nil }

        var distances: [(distance: Double, index: Int)] = []
        distances.reserveCapacity(count)
        for index in 0..<count {
            let dx = samples[index * 4] - point.x
            let dy = samples[index * 4 + 1] - point.y
            distances.append(((dx * dx + dy * dy).squareRoot(), index))
        }
        distances.sort { $0.distance < $1.distance }

        let quadratic = order >= 2
        let terms = quadratic ? 6 : 3

        var rows: [[Double]] = []
        var rhsX: [Double] = []
        var rhsY: [Double] = []
        var weightSum = 0.0
        var weightSquareSum = 0.0

        for entry in distances.prefix(min(max(neighbors, terms * 3), count)) {
            let u = samples[entry.index * 4] - point.x
            let v = samples[entry.index * 4 + 1] - point.y
            let scaled = entry.distance / max(bandwidth, 1e-6)
            let weight = exp(-0.5 * scaled * scaled)
            guard weight > 1e-4 else { continue }
            let root = weight.squareRoot()
            var row = [root, root * u, root * v]
            if quadratic { row += [root * u * u, root * u * v, root * v * v] }
            rows.append(row)
            rhsX.append(root * samples[entry.index * 4 + 2])
            rhsY.append(root * samples[entry.index * 4 + 3])
            weightSum += weight
            weightSquareSum += weight * weight
        }

        guard rows.count >= terms * 2, weightSum > 0 else { return nil }
        // Kish effective sample size: how many samples are really contributing, not how many
        // were considered. A handful of distant points on a 6-parameter fit is not support.
        let effective = weightSum * weightSum / max(weightSquareSum, 1e-9)
        guard effective >= minSupport else { return nil }

        let trace = rows.reduce(0.0) { $0 + $1[0] * $1[0] }
        let damping = max(trace * ridge, 1e-12).squareRoot()
        for column in 0..<terms {
            var row = [Double](repeating: 0, count: terms)
            row[column] = damping
            rows.append(row)
            rhsX.append(0)
            rhsY.append(0)
        }

        guard let coeffsX = Linalg.leastSquares(rows, rhsX),
              let coeffsY = Linalg.leastSquares(rows, rhsY),
              let rx = coeffsX.first, let ry = coeffsY.first,
              rx.isFinite, ry.isFinite else { return nil }

        // Confidence ramps in over twice the support floor, so the correction cannot switch on
        // abruptly at a neighbourhood boundary and make the pointer twitch.
        let confidence = min(1.0, effective / (2 * minSupport))
        return Vec2(rx, ry) * confidence
    }
}
