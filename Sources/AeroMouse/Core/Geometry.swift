import Foundation
import CoreGraphics

// MARK: - Vec2

struct Vec2: Equatable, Codable {
    var x: Double
    var y: Double

    init(_ x: Double = 0, _ y: Double = 0) {
        self.x = x
        self.y = y
    }

    init(_ p: CGPoint) {
        self.x = Double(p.x)
        self.y = Double(p.y)
    }

    static let zero = Vec2(0, 0)

    var length: Double { (x * x + y * y).squareRoot() }
    var isValid: Bool { x.isFinite && y.isFinite }
    var cg: CGPoint { CGPoint(x: x, y: y) }

    func distance(to other: Vec2) -> Double { (self - other).length }

    static func + (a: Vec2, b: Vec2) -> Vec2 { Vec2(a.x + b.x, a.y + b.y) }
    static func - (a: Vec2, b: Vec2) -> Vec2 { Vec2(a.x - b.x, a.y - b.y) }
    static func * (a: Vec2, s: Double) -> Vec2 { Vec2(a.x * s, a.y * s) }
    static func / (a: Vec2, s: Double) -> Vec2 { Vec2(a.x / s, a.y / s) }
    static prefix func - (a: Vec2) -> Vec2 { Vec2(-a.x, -a.y) }
    static func += (a: inout Vec2, b: Vec2) { a = a + b }
    static func *= (a: inout Vec2, s: Double) { a = a * s }

    static func lerp(_ a: Vec2, _ b: Vec2, _ t: Double) -> Vec2 {
        let c = min(max(t, 0), 1)
        return Vec2(a.x + (b.x - a.x) * c, a.y + (b.y - a.y) * c)
    }

    func clamped(to rect: RectD) -> Vec2 {
        guard rect.isValid else { return self }
        return Vec2(min(max(x, rect.minX), rect.maxX), min(max(y, rect.minY), rect.maxY))
    }
}

// MARK: - RectD (screen rectangles in CoreGraphics global coordinates)

struct RectD: Codable, Equatable {
    var x: Double
    var y: Double
    var width: Double
    var height: Double

    init(x: Double, y: Double, width: Double, height: Double) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }

    init(_ rect: CGRect) {
        self.x = Double(rect.origin.x)
        self.y = Double(rect.origin.y)
        self.width = Double(rect.width)
        self.height = Double(rect.height)
    }

    var minX: Double { x }
    var minY: Double { y }
    var maxX: Double { x + width }
    var maxY: Double { y + height }
    var mid: Vec2 { Vec2(x + width / 2, y + height / 2) }
    var cgRect: CGRect { CGRect(x: x, y: y, width: width, height: height) }
    var isValid: Bool { width > 1 && height > 1 && width.isFinite && height.isFinite }

    /// Screen normalized (0...1, origin top-left) -> global pixels.
    func denormalized(_ n: Vec2) -> Vec2 {
        Vec2(x + n.x * width, y + n.y * height)
    }

    /// Global pixels -> screen normalized (0...1, origin top-left).
    func normalized(_ p: Vec2) -> Vec2 {
        Vec2((p.x - x) / width, (p.y - y) / height)
    }
}

// MARK: - Dense linear algebra (small systems, no dependencies)

enum Linalg {
    /// Solves `a * x = b` for a square system using Gaussian elimination with partial pivoting.
    static func solve(_ aIn: [[Double]], _ bIn: [Double]) -> [Double]? {
        let n = aIn.count
        guard n > 0, bIn.count == n, aIn.allSatisfy({ $0.count == n }) else { return nil }
        var a = aIn
        var b = bIn

        for col in 0..<n {
            var pivot = col
            var best = abs(a[col][col])
            if col + 1 < n {
                for r in (col + 1)..<n where abs(a[r][col]) > best {
                    best = abs(a[r][col])
                    pivot = r
                }
            }
            if best < 1e-13 { return nil }
            if pivot != col {
                a.swapAt(pivot, col)
                b.swapAt(pivot, col)
            }
            let pv = a[col][col]
            if col + 1 < n {
                for r in (col + 1)..<n {
                    let factor = a[r][col] / pv
                    if factor == 0 { continue }
                    a[r][col] = 0
                    if col + 1 < n {
                        for c in (col + 1)..<n { a[r][c] -= factor * a[col][c] }
                    }
                    b[r] -= factor * b[col]
                }
            }
        }

        var x = [Double](repeating: 0, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            var sum = b[i]
            if i + 1 < n {
                for j in (i + 1)..<n { sum -= a[i][j] * x[j] }
            }
            let diag = a[i][i]
            if abs(diag) < 1e-13 { return nil }
            x[i] = sum / diag
        }
        guard x.allSatisfy({ $0.isFinite }) else { return nil }
        return x
    }
}

extension Linalg {
    /// Overdetermined least squares via Householder QR. Solving the normal equations would square
    /// the condition number, which destroys fits with correlated basis functions (RBF fields).
    static func leastSquares(_ aIn: [[Double]], _ bIn: [Double]) -> [Double]? {
        guard let first = aIn.first, !first.isEmpty, aIn.count >= first.count, bIn.count == aIn.count else {
            return nil
        }
        var a = aIn
        var b = bIn
        let m = a.count
        let n = first.count

        for k in 0..<n {
            var norm = 0.0
            for i in k..<m { norm += a[i][k] * a[i][k] }
            norm = norm.squareRoot()
            if norm < 1e-15 { continue }

            let alpha = a[k][k] > 0 ? -norm : norm
            var v = [Double](repeating: 0, count: m)
            for i in k..<m { v[i] = a[i][k] }
            v[k] -= alpha
            var vNormSquared = 0.0
            for i in k..<m { vNormSquared += v[i] * v[i] }
            if vNormSquared < 1e-24 { continue }

            for j in k..<n {
                var dot = 0.0
                for i in k..<m { dot += v[i] * a[i][j] }
                let factor = 2 * dot / vNormSquared
                if factor == 0 { continue }
                for i in k..<m { a[i][j] -= factor * v[i] }
            }
            var dotB = 0.0
            for i in k..<m { dotB += v[i] * b[i] }
            let factorB = 2 * dotB / vNormSquared
            for i in k..<m { b[i] -= factorB * v[i] }
        }

        var x = [Double](repeating: 0, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            var sum = b[i]
            if i + 1 < n {
                for j in (i + 1)..<n { sum -= a[i][j] * x[j] }
            }
            let diagonal = a[i][i]
            if abs(diagonal) < 1e-14 { return nil }
            x[i] = sum / diagonal
        }
        return x.allSatisfy { $0.isFinite } ? x : nil
    }
}

// MARK: - Mat3 / homography

struct Mat3: Codable, Equatable {
    var m: [Double]

    init(_ values: [Double]) {
        self.m = values.count == 9 ? values : [1, 0, 0, 0, 1, 0, 0, 0, 1]
    }

    static let identity = Mat3([1, 0, 0, 0, 1, 0, 0, 0, 1])

    func map(_ p: Vec2) -> Vec2 {
        let d = m[6] * p.x + m[7] * p.y + m[8]
        guard abs(d) > 1e-12 else { return .zero }
        return Vec2((m[0] * p.x + m[1] * p.y + m[2]) / d,
                    (m[3] * p.x + m[4] * p.y + m[5]) / d)
    }

    func times(_ o: Mat3) -> Mat3 {
        var r = [Double](repeating: 0, count: 9)
        for i in 0..<3 {
            for j in 0..<3 {
                var sum = 0.0
                for k in 0..<3 { sum += m[i * 3 + k] * o.m[k * 3 + j] }
                r[i * 3 + j] = sum
            }
        }
        return Mat3(r)
    }

    var determinant: Double {
        let a = m[0], b = m[1], c = m[2]
        let d = m[3], e = m[4], f = m[5]
        let g = m[6], h = m[7], i = m[8]
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    }

    var inverse: Mat3? {
        let a = m[0], b = m[1], c = m[2]
        let d = m[3], e = m[4], f = m[5]
        let g = m[6], h = m[7], i = m[8]
        let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        guard abs(det) > 1e-12 else { return nil }
        let s = 1.0 / det
        return Mat3([
            (e * i - f * h) * s, (c * h - b * i) * s, (b * f - c * e) * s,
            (f * g - d * i) * s, (a * i - c * g) * s, (c * d - a * f) * s,
            (d * h - e * g) * s, (b * g - a * h) * s, (a * e - b * d) * s,
        ])
    }

    /// Normalized (h33 == 1) copy, when possible.
    func normalizedCopy() -> Mat3 {
        let s = m[8]
        guard abs(s) > 1e-12 else { return self }
        return Mat3(m.map { $0 / s })
    }

    /// Least-squares homography fit with Hartley normalization (DLT + normal equations).
    static func fit(from src: [Vec2], to dst: [Vec2]) -> Mat3? {
        guard src.count == dst.count, src.count >= 4 else { return nil }
        let (srcN, srcT) = normalizePoints(src)
        let (dstN, dstT) = normalizePoints(dst)

        var ata = [[Double]](repeating: [Double](repeating: 0, count: 8), count: 8)
        var atb = [Double](repeating: 0, count: 8)

        for i in 0..<src.count {
            let x = srcN[i].x, y = srcN[i].y
            let u = dstN[i].x, v = dstN[i].y
            let r1: [Double] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
            let r2: [Double] = [0, 0, 0, x, y, 1, -v * x, -v * y]
            for a in 0..<8 {
                atb[a] += r1[a] * u + r2[a] * v
                for b in 0..<8 { ata[a][b] += r1[a] * r1[b] + r2[a] * r2[b] }
            }
        }

        let trace = (0..<8).reduce(0.0) { $0 + ata[$1][$1] }
        let lambda = max(trace * 1e-12, 1e-15)
        for a in 0..<8 { ata[a][a] += lambda }

        guard let h = Linalg.solve(ata, atb) else { return nil }
        let hn = Mat3([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1])
        guard let dstTInv = dstT.inverse else { return nil }
        let composed = dstTInv.times(hn).times(srcT)
        guard composed.determinant.isFinite else { return nil }
        return composed.normalizedCopy()
    }

    private static func normalizePoints(_ pts: [Vec2]) -> ([Vec2], Mat3) {
        let n = Double(pts.count)
        guard n > 0 else { return (pts, .identity) }
        let cx = pts.reduce(0.0) { $0 + $1.x } / n
        let cy = pts.reduce(0.0) { $0 + $1.y } / n
        let meanDist = pts.reduce(0.0) { acc, p in
            let dx = p.x - cx, dy = p.y - cy
            return acc + (dx * dx + dy * dy).squareRoot()
        } / n
        let scale = meanDist > 1e-9 ? (2.0.squareRoot() / meanDist) : 1.0
        let t = Mat3([scale, 0, -scale * cx, 0, scale, -scale * cy, 0, 0, 1])
        let normalized = pts.map { Vec2(($0.x - cx) * scale, ($0.y - cy) * scale) }
        return (normalized, t)
    }
}
