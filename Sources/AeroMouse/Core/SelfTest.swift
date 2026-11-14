import Foundation
import Vision

struct SelfTestRunner {
    private(set) var checks = 0
    private(set) var failures = 0

    mutating func check(_ name: String, _ passed: Bool, _ detail: String = "") {
        checks += 1
        if !passed { failures += 1 }
        let suffix = detail.isEmpty ? "" : "  [\(detail)]"
        print("\(passed ? "PASS" : "FAIL")  \(name)\(suffix)")
    }
}

enum SelfTest {
    static func runAll() -> Int {
        var runner = SelfTestRunner()
        print("AeroMouse self-test\n")
        runChecks(&runner)
        print("\n\(runner.checks - runner.failures)/\(runner.checks) checks passed")
        return runner.failures == 0 ? 0 : 1
    }

    static func summary() -> (checks: Int, failures: Int) {
        var runner = SelfTestRunner()
        runChecks(&runner)
        return (runner.checks, runner.failures)
    }

    private static func runChecks(_ runner: inout SelfTestRunner) {
        geometry(&runner)
        regression(&runner)
        neural(&runner)
        smoothing(&runner)
        gestures(&runner)
        lowLight(&runner)
        landmarkFiltering(&runner)
        handAnchor(&runner)
        cameraModes(&runner)
        localCorrection(&runner)
        calibrationEndToEnd(&runner)
    }

    // MARK: Geometry

    private static func geometry(_ runner: inout SelfTestRunner) {
        let truth = Mat3([1.7, 0.21, -0.33, -0.12, 1.55, 0.44, 0.35, -0.28, 1.0])
        var sources: [Vec2] = []
        var targets: [Vec2] = []
        var rng = SplitMix64(seed: 11)
        for _ in 0..<40 {
            let p = Vec2(rng.nextDouble(), rng.nextDouble())
            sources.append(p)
            targets.append(truth.map(p))
        }

        guard let fitted = Mat3.fit(from: sources, to: targets) else {
            runner.check("homography fit returns a matrix", false)
            return
        }
        var maxError = 0.0
        for i in 0..<sources.count {
            maxError = max(maxError, fitted.map(sources[i]).distance(to: targets[i]))
        }
        runner.check("homography recovers a known projection", maxError < 1e-6,
                     String(format: "max residual %.2e", maxError))

        if let inverse = fitted.inverse {
            let p = Vec2(0.37, 0.62)
            let roundTrip = inverse.map(fitted.map(p))
            runner.check("homography inverse round trip", roundTrip.distance(to: p) < 1e-6,
                         String(format: "error %.2e", roundTrip.distance(to: p)))
        } else {
            runner.check("homography inverse exists", false)
        }

        var noisy: [Vec2] = []
        for i in 0..<sources.count {
            noisy.append(targets[i] + Vec2((rng.nextDouble() - 0.5) * 0.002,
                                           (rng.nextDouble() - 0.5) * 0.002))
        }
        if let noisyFit = Mat3.fit(from: sources, to: noisy) {
            var total = 0.0
            for i in 0..<sources.count { total += noisyFit.map(sources[i]).distance(to: targets[i]) }
            let mean = total / Double(sources.count)
            runner.check("homography is robust to sensor noise", mean < 0.002,
                         String(format: "mean error %.4f", mean))
        } else {
            runner.check("homography fits noisy data", false)
        }
    }

    // MARK: Regression

    private static func regression(_ runner: inout SelfTestRunner) {
        let truth = { (p: Vec2) -> Vec2 in
            Vec2(0.9 * p.x + 0.1 * p.y + 0.25 * p.x * p.x - 0.18 * p.x * p.y,
                 0.05 * p.x + 0.95 * p.y + 0.31 * p.y * p.y + 0.12 * p.x * p.y)
        }
        var samples: [CalibrationSample] = []
        var rng = SplitMix64(seed: 23)
        var group = 0
        for _ in 0..<60 {
            let p = Vec2(rng.nextDouble(), rng.nextDouble())
            samples.append(CalibrationSample(source: p, target: truth(p), weight: 1,
                                             group: group, pose: PointerPose(), gaze: nil))
            group += 1
        }

        if let poly = PolynomialMap.fit(samples, degree: 2) {
            var maxError = 0.0
            for s in samples { maxError = max(maxError, poly.map(s.source).distance(to: s.target)) }
            runner.check("quadratic regression recovers a quadratic field", maxError < 3e-5,
                         String(format: "max residual %.2e (%.3f px)", maxError, maxError * 1600))
        } else {
            runner.check("quadratic regression fits", false)
        }

        // Model selection must prefer the regression over a plane when the field is curved.
        let orderingScreen = RectD(x: 0, y: 0, width: 1600, height: 1000)
        let quadraticCV = CalibrationCV.run(samples: samples, folds: 5) { training in
            let map = MapFitter.fit(kind: .poly2, samples: training)
            return { s in map?.raw(s.source, pose: s.pose) ?? s.source }
        }
        let planeCV = CalibrationCV.run(samples: samples, folds: 5) { training in
            let map = MapFitter.fit(kind: .homography, samples: training)
            return { s in map?.raw(s.source, pose: s.pose) ?? s.source }
        }
        let quadraticMetrics = CalibrationCV.metrics(quadraticCV, screen: orderingScreen)
        let planeMetrics = CalibrationCV.metrics(planeCV, screen: orderingScreen)
        runner.check("model selection prefers regression on a curved field",
                     quadraticMetrics.median < planeMetrics.median,
                     String(format: "%.3f px vs %.3f px", quadraticMetrics.median, planeMetrics.median))

        // Pose-conditioned regression must exactly recover a depth/roll-dependent linear field.
        var posed: [CalibrationSample] = []
        var poseRng = SplitMix64(seed: 77)
        for i in 0..<80 {
            let source = Vec2(poseRng.nextDouble(), poseRng.nextDouble())
            let pose = PointerPose(handScale: 0.12 + poseRng.nextDouble() * 0.07,
                                   roll: (poseRng.nextDouble() - 0.5) * 0.6,
                                   spread: 1 + poseRng.nextDouble() * 0.2)
            let b = PoseRegressionMap.basis(source, pose: pose, referenceScale: 0.155)
            let tx = 0.1 + 0.8 * b[1] + 0.05 * b[3] + 0.2 * b[6]
            let ty = 0.05 + 0.75 * b[2] + 0.04 * b[4] + 0.15 * b[7]
            posed.append(CalibrationSample(source: source, target: Vec2(tx, ty), weight: 1,
                                           group: i, pose: pose, gaze: nil))
        }
        if let poseMap = PoseRegressionMap.fit(posed, referenceScale: 0.155) {
            var maxPoseError = 0.0
            for s in posed {
                maxPoseError = max(maxPoseError, poseMap.map(s.source, pose: s.pose).distance(to: s.target))
            }
            runner.check("pose regression recovers a depth-dependent field", maxPoseError < 1e-6,
                         String(format: "max residual %.2e", maxPoseError))
        } else {
            runner.check("pose regression fits", false)
        }

        let screen = RectD(x: 0, y: 0, width: 1600, height: 1000)
        let predictions = CalibrationCV.run(samples: samples, folds: 4) { _ in
            { _ in Vec2(0.5, 0.5) }
        }
        let metrics = CalibrationCV.metrics(predictions, screen: screen)
        runner.check("grouped cross-validation reports real error", metrics.rms > 50,
                     String(format: "constant model RMS %.1f px", metrics.rms))
        runner.check("cross-validation covers every sample", predictions.count == samples.count,
                     "\(predictions.count) of \(samples.count)")
    }

    // MARK: Neural

    private static func neural(_ runner: inout SelfTestRunner) {
        var rng = SplitMix64(seed: 5)
        var inputs: [[Double]] = []
        var outputs: [[Double]] = []
        for _ in 0..<400 {
            let x = rng.nextDouble() * 2 - 1
            let y = rng.nextDouble() * 2 - 1
            inputs.append([x, y])
            outputs.append([0.6 * x - 0.35 * y + 0.4 * x * y + 0.15 * sin(2.5 * x)])
        }
        let scaler = StandardScaler.fit(inputs)
        let scaled = inputs.map { scaler.transform($0) }

        var net = MLP(sizes: [2, 16, 16, 1], seed: 3)
        let config = TrainingConfig(epochs: 900, batchSize: 32, learningRate: 0.02,
                                    weightDecay: 1e-6, validationFraction: 0.2, patience: 300, seed: 9)
        let report = MLPTrainer.train(&net, inputs: scaled, outputs: outputs, config: config)

        var worst = 0.0
        var total = 0.0
        for i in 0..<scaled.count {
            let error = abs(net.predict(scaled[i])[0] - outputs[i][0])
            total += error
            worst = max(worst, error)
        }
        let meanError = total / Double(scaled.count)
        runner.check("neural net learns a non-linear surface", meanError < 0.02,
                     String(format: "mean %.4f, worst %.4f", meanError, worst))
        runner.check("training reports a validation loss", report.validationLoss.isFinite && report.bestEpoch > 0,
                     "best epoch \(report.bestEpoch)")

        let extreme = net.predict([40, -40])
        runner.check("neural net is numerically safe", extreme.allSatisfy { $0.isFinite },
                     String(format: "output %.3f", extreme.first ?? .nan))

        var broken = MLP(sizes: [2, 8, 1], seed: 1)
        let empty = MLPTrainer.train(&broken, inputs: [], outputs: [], config: config)
        runner.check("trainer rejects empty data", !empty.trainLoss.isFinite, "")
    }

    // MARK: Smoothing

    private static func smoothing(_ runner: inout SelfTestRunner) {
        let bounds = RectD(x: 0, y: 0, width: 1600, height: 1000)
        let pipeline = PointerPipeline()
        let config = SmoothingConfig()
        var rng = SplitMix64(seed: 31)
        let base = Vec2(820, 500)
        var inputs: [Vec2] = []
        var outputs: [Vec2] = []

        for i in 0..<200 {
            let time = Double(i) / 60.0
            let noise = Vec2((rng.nextDouble() - 0.5) * 7, (rng.nextDouble() - 0.5) * 7)
            let point = base + noise
            inputs.append(point)
            if let out = pipeline.process(modelPoint: point, time: time, bounds: bounds, config: config), i > 40 {
                outputs.append(out.point)
            }
        }

        func rms(_ points: [Vec2]) -> Double {
            guard !points.isEmpty else { return 0 }
            let mean = points.reduce(Vec2.zero, +) / Double(points.count)
            let variance = points.reduce(0.0) { $0 + pow($1.distance(to: mean), 2) } / Double(points.count)
            return variance.squareRoot()
        }

        let before = rms(Array(inputs.dropFirst(40)))
        let after = rms(outputs)
        runner.check("One Euro filter suppresses tremor", after < 1.25 && after < before * 0.7,
                     String(format: "%.2f px → %.2f px", before, after))

        // A single-frame teleport must not drag the cursor across the screen.
        let jumpPipeline = PointerPipeline()
        var held = true
        for i in 0..<10 {
            let point = i < 6 ? Vec2(400, 400) : Vec2(1200, 800)
            if let out = jumpPipeline.process(modelPoint: point, time: Double(i) / 60.0,
                                              bounds: bounds, config: config) {
                if i == 6 { held = out.point.distance(to: Vec2(400, 400)) < 30 }
            }
        }
        runner.check("outlier gate holds through a bad frame", held, "")

        // Precision mode must attenuate slow motion but not fast motion.
        runner.check("precision gain blends to full speed when moving fast",
                     PointerPipeline.selfTestsGainAtSpeed(400) > 0.9, "")
    }

    // MARK: Gestures

    private static func gestures(_ runner: inout SelfTestRunner) {
        let recognizer = GestureRecognizer()
        var config = GestureConfig()
        config.minDownFrames = 2
        config.minUpFrames = 2

        var events: [PointerEvent] = []
        var time = 0.0
        let approach = stride(from: 0.90, through: 0.30, by: -0.02).map { $0 }
            + stride(from: 0.32, through: 0.90, by: 0.02).map { $0 }
        for value in approach {
            time += 1.0 / 60.0
            let out = recognizer.update(pinchIndex: value, pinchMiddle: 1.4, fistRatio: 0.1,
                                        time: time, config: config)
            events.append(contentsOf: out.events)
        }
        let downs = events.filter { if case .leftDown = $0 { return true }; return false }.count
        let ups = events.filter { $0 == .leftUp }.count
        runner.check("pinch hover produces exactly one click", downs == 1 && ups == 1,
                     "\(downs) down, \(ups) up")

        // Two quick pinches inside the double-click window must carry clickState 2.
        let dc = GestureRecognizer()
        var dcEvents: [PointerEvent] = []
        var t = 0.0
        for _ in 0..<2 {
            for value in [0.9, 0.9, 0.30, 0.30, 0.30, 0.90, 0.90, 0.90] {
                t += 1.0 / 60.0
                dcEvents.append(contentsOf: dc.update(pinchIndex: value, pinchMiddle: 1.4,
                                                      fistRatio: 0.1, time: t, config: config).events)
            }
        }
        let clickStates = dcEvents.compactMap { event -> Int? in
            if case .leftDown(let state) = event { return state }
            return nil
        }
        runner.check("double pinch escalates click state", clickStates.last == 2,
                     "states \(clickStates)")

        // A closed fist must freeze and release any held button (no stuck drags).
        let fist = GestureRecognizer()
        var fistEvents: [PointerEvent] = []
        var mt = 0.0
        for value in [0.3, 0.3, 0.3, 0.9, 0.9] {
            mt += 1.0 / 60.0
            fistEvents.append(contentsOf: fist.update(pinchIndex: value, pinchMiddle: 1.4,
                                                      fistRatio: value > 0.85 ? 0.95 : 0.1,
                                                      time: mt, config: config).events)
        }
        runner.check("fist releases a held click", fistEvents.contains(.leftUp), "\(fistEvents.count) events")
    }

    // MARK: Low light

    private static func lowLight(_ runner: inout SelfTestRunner) {
        let bright = LowLightPolicy.decision(forLuma: 0.72)
        let dim = LowLightPolicy.decision(forLuma: 0.30)
        let dark = LowLightPolicy.decision(forLuma: 0.10)

        runner.check("bright frames pass through untouched",
                     !bright.enhance && bright.targetFPS == 60 && bright.temporalBlend == 0, "")
        runner.check("dim frames get GPU enhancement",
                     dim.enhance && dim.shadowLift > 0 && dim.temporalBlend > 0,
                     String(format: "lift %.2f, blend %.2f", dim.shadowLift, dim.temporalBlend))
        runner.check("low light lowers the frame rate for longer exposure",
                     dark.targetFPS == 30 && dim.targetFPS == 30, "\(Int(dark.targetFPS)) fps")
        runner.check("shadow lift grows as light falls",
                     dark.shadowLift > dim.shadowLift && dim.shadowLift > bright.shadowLift,
                     String(format: "%.2f > %.2f", dark.shadowLift, dim.shadowLift))
        runner.check("temporal averaging grows as light falls",
                     dark.temporalBlend > dim.temporalBlend, "")
    }

    // MARK: Landmark filtering

    private static func landmarkFiltering(_ runner: inout SelfTestRunner) {
        let smoother = LandmarkSmoother()
        var rng = SplitMix64(seed: 61)

        let joints: [VNHumanHandPoseObservation.JointName] = [
            .wrist, .middleMCP, .indexMCP, .indexDIP, .indexTip,
            .thumbTip, .middleTip, .ringTip, .littleTip, .ringMCP, .littleMCP,
        ]
        var truth: [VNHumanHandPoseObservation.JointName: Vec2] = [:]
        for (index, joint) in joints.enumerated() {
            truth[joint] = Vec2(0.30 + Double(index % 4) * 0.06, 0.40 + Double(index / 4) * 0.06)
        }

        var raw: [Double] = []
        var filtered: [Double] = []

        for frame in 0..<140 {
            let time = Double(frame) / 60.0
            var noisy: [VNHumanHandPoseObservation.JointName: Vec2] = [:]
            var confidences: [VNHumanHandPoseObservation.JointName: Double] = [:]
            for (joint, point) in truth {
                let noise = Vec2((rng.nextDouble() - 0.5) * 0.008,
                                 (rng.nextDouble() - 0.5) * 0.008)
                noisy[joint] = point + noise
                confidences[joint] = 0.90
            }
            let smoothed = smoother.smooth(noisy, confidences: confidences, time: time)
            if frame > 20 {
                raw.append(noisy[.indexTip]?.x ?? 0)
                filtered.append(smoothed[.indexTip]?.x ?? 0)
            }
        }

        func standardDeviation(_ values: [Double]) -> Double {
            guard !values.isEmpty else { return 0 }
            let mean = values.reduce(0, +) / Double(values.count)
            let variance = values.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / Double(values.count)
            return variance.squareRoot()
        }

        let before = standardDeviation(raw)
        let after = standardDeviation(filtered)
        runner.check("landmark filter suppresses joint noise", after < before * 0.6,
                     String(format: "%.4f → %.4f normalized", before, after))

        // A joint that disappears must not linger forever and fake a pinch distance.
        let stale = smoother.smooth([.wrist: truth[.wrist] ?? .zero],
                                    confidences: [.wrist: 0.9],
                                    time: 10.0)
        runner.check("stale joints are dropped", stale[.indexTip] == nil, "")

        // A weak, implausible jump must not be followed out of nowhere.
        let jumpSmoother = LandmarkSmoother()
        _ = jumpSmoother.smooth([.indexTip: Vec2(0.30, 0.30)],
                                confidences: [.indexTip: 0.95],
                                time: 1.00)
        let damped = jumpSmoother.smooth([.indexTip: Vec2(0.90, 0.90)],
                                         confidences: [.indexTip: 0.20],
                                         time: 1.02)
        let landed = damped[.indexTip] ?? Vec2(9, 9)
        runner.check("weak detections cannot teleport a joint",
                     landed.distance(to: Vec2(0.30, 0.30)) < 0.25,
                     String(format: "drift %.3f", landed.distance(to: Vec2(0.30, 0.30))))
    }

    // MARK: Anchor

    private static func handAnchor(_ runner: inout SelfTestRunner) {
        // A plausible pointing hand in aspect-corrected image space. The tip and its last joint
        // swing toward the thumb as the finger curls into a pinch.
        func joints(curl: Double) -> [VNHumanHandPoseObservation.JointName: Vec2] {
            [
                .wrist: Vec2(0.62, 0.40),
                .middleMCP: Vec2(0.62, 0.60),
                .indexMCP: Vec2(0.56, 0.63),
                .indexPIP: Vec2(0.55, 0.75),
                .indexDIP: Vec2(0.57 - 0.05 * curl, 0.80 - 0.07 * curl),
                .indexTip: Vec2(0.56 - 0.10 * curl, 0.88 - 0.14 * curl),
                .thumbTip: Vec2(0.45, 0.62),
                .middleTip: Vec2(0.64, 0.92),
                .ringTip: Vec2(0.72, 0.90),
                .littleTip: Vec2(0.79, 0.85),
                .ringMCP: Vec2(0.70, 0.60),
                .littleMCP: Vec2(0.75, 0.57),
            ]
        }

        guard let open = HandFrame.build(points: joints(curl: 0), confidences: [:]),
              let pinched = HandFrame.build(points: joints(curl: 1), confidences: [:]) else {
            runner.check("hand frame builds from a pointing hand", false, "nil")
            return
        }
        runner.check("hand frame builds from a pointing hand", true, "")

        let anchorTravel = open.anchor.distance(to: pinched.anchor)
        let tipTravel = Vec2(0.56, 0.88).distance(to: Vec2(0.46, 0.74))
        runner.check("pinch does not move the pointer anchor",
                     anchorTravel < tipTravel * 0.35,
                     String(format: "anchor %.3f vs fingertip %.3f", anchorTravel, tipTravel))

        // The outer fingers are routinely lost at the edge of frame; the pointer must survive.
        var partial = joints(curl: 0)
        partial[.ringTip] = nil
        partial[.littleTip] = nil
        partial[.ringMCP] = nil
        partial[.littleMCP] = nil
        partial[.middleTip] = nil
        let sparse = HandFrame.build(points: partial, confidences: [:])
        runner.check("tracking survives a partially visible hand",
                     sparse != nil && (sparse?.pinchMiddle.isInfinite ?? false), "")
    }

    // MARK: Camera modes

    private static func cameraModes(_ runner: inout SelfTestRunner) {
        let hd = CameraCapture.score(width: 1280, height: 720, maxRate: 30, preferHighResolution: true)
        let vga = CameraCapture.score(width: 640, height: 480, maxRate: 60, preferHighResolution: true)
        runner.check("accurate mode prefers 720p over a 60 fps sensor mode", hd > vga,
                     String(format: "%.1f vs %.1f", hd, vga))

        let fastHD = CameraCapture.score(width: 1280, height: 720, maxRate: 30, preferHighResolution: false)
        let fastVGA = CameraCapture.score(width: 640, height: 480, maxRate: 60, preferHighResolution: false)
        runner.check("fast mode prefers the small high-rate sensor mode", fastVGA > fastHD,
                     String(format: "%.1f vs %.1f", fastVGA, fastHD))

        let tiny = CameraCapture.score(width: 320, height: 240, maxRate: 60, preferHighResolution: true)
        runner.check("modes below 640×480 are rejected", tiny == -Double.infinity, "")
    }

    // MARK: Local residual

    private static func localCorrection(_ runner: inout SelfTestRunner) {
        var rng = SplitMix64(seed: 909)
        let screen = RectD(x: 0, y: 0, width: 1920, height: 1080)
        let pose = PointerPose(handScale: 0.15, roll: 0, spread: 1.1)

        func error(_ candidate: Vec2, _ expected: Vec2) -> Double {
            let dx = (candidate.x - expected.x) * screen.width
            let dy = (candidate.y - expected.y) * screen.height
            return (dx * dx + dy * dy).squareRoot()
        }

        func median(_ values: [Double]) -> Double {
            guard !values.isEmpty else { return 0 }
            let sorted = values.sorted()
            return sorted[sorted.count / 2]
        }

        // A truth that bends between the calibration dots. No single quadratic can follow it, which
        // is the mid-cell error a user feels but a fit measured only at the dots never sees.
        func truth(_ source: Vec2) -> Vec2 {
            let u = (source.x - 0.42) / 0.24
            let v = (source.y - 0.30) / 0.30
            let ripple = 0.030 * sin(u * 7.5) * cos(v * 6.0)
            return Vec2(min(max(u * 0.92 + 0.04 + ripple, 0), 1),
                        min(max(v * 0.90 + 0.05 - ripple * 0.7, 0), 1))
        }

        var samples: [CalibrationSample] = []
        for row in 0..<5 {
            for column in 0..<5 {
                for _ in 0..<3 {
                    let source = Vec2(0.42 + Double(column) * 0.060 + (rng.nextDouble() - 0.5) * 0.010,
                                      0.30 + Double(row) * 0.075 + (rng.nextDouble() - 0.5) * 0.010)
                    samples.append(CalibrationSample(source: source,
                                                     target: truth(source),
                                                     weight: 1,
                                                     group: row * 5 + column,
                                                     pose: pose,
                                                     gaze: nil))
                }
            }
        }

        guard let fitted = MapFitter.fit(kind: .poly2, samples: samples,
                                        maxCenters: 220, referenceScale: 0.15) else {
            runner.check("global fit covers the warped truth", false, "nil")
            return
        }

        var pairs: [(mapped: Vec2, target: Vec2)] = []
        for sample in samples {
            guard let mapped = fitted.raw(sample.source, pose: sample.pose) else { continue }
            pairs.append((mapped, sample.target))
        }

        // Sweep the two knobs so the shipped defaults come from measurement, not taste.
        var tuned = ""
        var tuning = [(bandwidth: 0.14, order: 2, neighbors: 40),
                      (bandwidth: 0.16, order: 2, neighbors: 40),
                      (bandwidth: 0.16, order: 2, neighbors: 60),
                      (bandwidth: 0.18, order: 2, neighbors: 40)]
        for config in tuning {
            guard let candidate = LocalResidual(predictions: pairs,
                                                bandwidth: config.bandwidth,
                                                order: config.order,
                                                neighbors: config.neighbors) else { continue }
            var errors: [Double] = []
            var probe = SplitMix64(seed: 5150)
            for _ in 0..<400 {
                let source = Vec2(0.40 + probe.nextDouble() * 0.27, 0.28 + probe.nextDouble() * 0.33)
                guard let base = fitted.raw(source, pose: pose) else { continue }
                let corrected = candidate.residual(at: base).map { base + $0 } ?? base
                errors.append(error(corrected, truth(source)))
            }
            tuned += String(format: "%g/o%d=%.1f ", config.bandwidth, config.order, median(errors))
        }

        guard let local = LocalResidual(predictions: pairs) else {
            runner.check("local residual stage builds", false, "nil")
            return
        }
        runner.check("local residual stage builds", true, "\(local.count) entries · \(tuned)")

        // Probe between the dots, where accuracy actually matters.
        var before: [Double] = []
        var after: [Double] = []
        for _ in 0..<400 {
            let source = Vec2(0.40 + rng.nextDouble() * 0.27, 0.28 + rng.nextDouble() * 0.33)
            guard let base = fitted.raw(source, pose: pose) else { continue }
            let expected = truth(source)
            before.append(error(base, expected))
            let corrected = local.residual(at: base).map { base + $0 } ?? base
            after.append(error(corrected, expected))
        }

        let baseline = median(before)
        let corrected = median(after)
        runner.check("local stage cuts mid-cell error",
                     corrected < baseline * 0.72,
                     String(format: "%.1f px → %.1f px (%.2f×)", baseline, corrected,
                            baseline / max(corrected, 1e-6)))
    }

    // MARK: End to end

    private static func calibrationEndToEnd(_ runner: inout SelfTestRunner) {
        let screen = RectD(x: 0, y: 0, width: 1600, height: 1000)
        let plan = CalibrationPlan.standard()
        let targets = plan.targets()

        // Simulated physical mapping: screen target -> what the tracker reports. The bend terms
        // and pose terms are exactly the distortion a pure homography cannot represent.
        func anchor(for target: Vec2, scale: Double, roll: Double, rng: inout SplitMix64) -> Vec2 {
            let nx = target.x - 0.5
            let ny = target.y - 0.5
            // Moving closer to the camera stretches the projected hand about the image centre:
            // a multiplicative, depth-dependent gain that no homography can represent.
            let zoom = 1 + (scale - 0.155) * 2.2
            // Gentle palm / lens curvature.
            let bendX = 0.010 * sin(2.3 * target.y + 0.6) + 0.006 * sin(3.1 * target.x)
            let bendY = 0.009 * sin(2.1 * target.x + 0.3) + 0.005 * sin(3.4 * target.y)
            let rollTerm = 0.006 * sin(roll)
            let noiseX = (rng.nextDouble() - 0.5) * 0.0025
            let noiseY = (rng.nextDouble() - 0.5) * 0.0025
            return Vec2(0.5 + nx * 0.82 * zoom + bendX + rollTerm + noiseX,
                        0.5 + ny * 0.82 * zoom + bendY + noiseY)
        }

        var rng = SplitMix64(seed: 99)
        var samples: [CalibrationSample] = []
        let total = targets.count * 3
        for (index, target) in targets.enumerated() {
            for sampleIndex in 0..<3 {
                // Hand distance and roll drift through the sequence, the way they do in reality.
                let phase = Double(index * 3 + sampleIndex) / Double(total)
                let scale = 0.155 + 0.030 * sin(2 * .pi * phase * 1.5) + (rng.nextDouble() - 0.5) * 0.006
                let roll = 0.30 * sin(2 * .pi * phase * 0.9) + (rng.nextDouble() - 0.5) * 0.05
                let point = anchor(for: target, scale: scale, roll: roll, rng: &rng)
                samples.append(CalibrationSample(
                    source: point,
                    target: target,
                    weight: 2.0,
                    group: plan.groupIndex(for: index),
                    pose: PointerPose(handScale: scale, roll: roll, spread: 1 + rng.nextDouble() * 0.25),
                    gaze: nil
                ))
            }
        }
        _ = total

        guard let model = Calibrator.build(samples: samples, screen: screen) else {
            runner.check("end-to-end calibration builds a model", false)
            return
        }

        let stats = model.stats
        runner.check("end-to-end calibration builds a model", !model.maps.isEmpty && stats.samples == samples.count,
                     "\(stats.samples) samples, \(stats.chosen.joined(separator: "+"))")

        let improvement = stats.medianBefore > 0 ? stats.medianBefore / max(stats.medianAfter, 0.001) : 0
        runner.check("stack beats the best single geometric fit", improvement > 2.0,
                     String(format: "%.1f px → %.1f px (%.1f×)", stats.medianBefore, stats.medianAfter, improvement))
        runner.check("calibration reaches sub-6 px median on held-out cells", stats.medianAfter < 6,
                     String(format: "median %.2f px, p95 %.2f px", stats.medianAfter, stats.p95After))

        // The calibration must survive a serialization round trip (it is persisted to disk).
        if let data = try? JSONEncoder().encode(model),
           let restored = try? JSONDecoder().decode(PointerModel.self, from: data) {
            let probe = samples[0]
            let a = model.normalized(for: probe.source, pose: probe.pose)
            let b = restored.normalized(for: probe.source, pose: probe.pose)
            runner.check("model round trips through JSON", a.distance(to: b) < 1e-9,
                         String(format: "delta %.2e", a.distance(to: b)))
        } else {
            runner.check("model round trips through JSON", false)
        }

        // Extrapolation safety: far outside the calibrated area the correction must fade out.
        let wild = model.normalized(for: Vec2(-3, 4), pose: PointerPose(handScale: 0.4, roll: 2, spread: 4))
        runner.check("model stays finite far outside calibration", wild.isValid, "")
    }
}
