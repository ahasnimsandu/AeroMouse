import Foundation

// MARK: - Plan

struct CalibrationPlan {
    var columns: Int = 5
    var rows: Int = 5
    var passes: Int = 3
    var edgeInset: Double = 0.05
    var jitter: Double = 0.012
    var seed: UInt64 = 42

    static func quick() -> CalibrationPlan {
        CalibrationPlan(columns: 3, rows: 3, passes: 3)
    }

    static func standard() -> CalibrationPlan {
        CalibrationPlan(columns: 5, rows: 5, passes: 3)
    }

    static func precise() -> CalibrationPlan {
        CalibrationPlan(columns: 7, rows: 7, passes: 3)
    }

    /// Ordered list of screen-normalized targets. Every pass revisits each cell with a small
    /// random offset, so samples spread across the plane instead of piling onto 25 pixels.
    func targets() -> [Vec2] {
        var rng = SplitMix64(seed: seed)
        var out: [Vec2] = []
        let cols = max(columns, 2)
        let rows = max(rows, 2)
        out.reserveCapacity(cols * rows * max(passes, 1))
        for _ in 0..<max(passes, 1) {
            for r in 0..<rows {
                for c in 0..<cols {
                    let u = edgeInset + (1 - 2 * edgeInset) * Double(c) / Double(cols - 1)
                    let v = edgeInset + (1 - 2 * edgeInset) * Double(r) / Double(rows - 1)
                    let jx = (rng.nextDouble() * 2 - 1) * jitter
                    let jy = (rng.nextDouble() * 2 - 1) * jitter
                    out.append(Vec2(min(max(u + jx, 0.02), 0.98),
                                    min(max(v + jy, 0.02), 0.98)))
                }
            }
        }
        return out
    }

    /// Which physical cell a target belongs to. Used to group cross-validation folds.
    func groupIndex(for targetIndex: Int) -> Int {
        let perPass = max(columns, 2) * max(rows, 2)
        return perPass > 0 ? targetIndex % perPass : targetIndex
    }
}

// MARK: - Residual network

enum ResidualNet {
    struct Bundle {
        var mlp: MLP
        var scaler: StandardScaler
        var trust: TrustGate
        var report: TrainingReport
    }

    /// Learns what the geometric mappings cannot express: residual fingerprint of depth, wrist
    /// roll and finger spread. Trained purely on this user's calibration samples.
    static func fit(samples: [CalibrationSample],
                    mapper: (Vec2, PointerPose) -> Vec2,
                    referenceScale: Double,
                    epochs: Int) -> Bundle? {
        guard samples.count >= 24 else { return nil }

        var features: [[Double]] = []
        var targets: [[Double]] = []
        features.reserveCapacity(samples.count)
        targets.reserveCapacity(samples.count)

        for sample in samples {
            let mapped = mapper(sample.source, sample.pose)
            guard mapped.isValid else { continue }
            features.append(PointerFeatures.vector(mapped: mapped, pose: sample.pose, referenceScale: referenceScale))
            targets.append([sample.target.x - mapped.x, sample.target.y - mapped.y])
        }
        guard features.count >= 24 else { return nil }

        let scaler = StandardScaler.fit(features)
        let scaled = features.map { scaler.transform($0) }

        var net = MLP(sizes: [PointerFeatures.dimension, 24, 24, 2], seed: 17)
        let config = TrainingConfig(epochs: epochs,
                                    batchSize: 32,
                                    learningRate: 0.008,
                                    weightDecay: 1e-5,
                                    validationFraction: 0.25,
                                    patience: 220,
                                    seed: 21)
        let report = MLPTrainer.train(&net, inputs: scaled, outputs: targets, config: config)
        guard report.validationLoss.isFinite else { return nil }

        let step = max(1, scaled.count / 400)
        let gatePoints = Swift.stride(from: 0, to: scaled.count, by: step).map { scaled[$0] }
        let trust = TrustGate(points: gatePoints, inner: 1.0, outer: 3.2)
        return Bundle(mlp: net, scaler: scaler, trust: trust, report: report)
    }
}

// MARK: - Calibrator

enum Calibrator {
    struct Options {
        var folds: Int = 5
        var maxCenters: Int = 220
        var useResidualNet: Bool = true
        var minSamples: Int = 16
    }

    /// Fits every candidate mapping, cross-validates them on held-out target cells, blends the
    /// good ones by error, and only keeps the neural correction if it measurably helps.
    static func build(samples: [CalibrationSample],
                      screen: RectD,
                      options: Options = Options()) -> PointerModel? {
        let usable = samples.filter { $0.source.isValid && $0.target.isValid && $0.weight > 0 }
        guard usable.count >= options.minSamples, screen.isValid else { return nil }

        let groupCount = Set(usable.map { $0.group }).count
        guard groupCount >= 4 else { return nil }
        let folds = max(2, min(options.folds, groupCount))

        let referenceScale = medianScale(usable)

        // 1. Independent grouped CV for every candidate mapping.
        var candidates: [(map: FittedMap, metrics: ErrorMetrics)] = []
        for kind in FittedMap.Kind.allCases {
            let predictions = CalibrationCV.run(samples: usable, folds: folds) { training in
                let fitted = MapFitter.fit(kind: kind, samples: training,
                                          maxCenters: options.maxCenters,
                                          referenceScale: referenceScale)
                return { sample in fitted?.raw(sample.source, pose: sample.pose) ?? sample.source }
            }
            guard !predictions.isEmpty else { continue }
            let metrics = CalibrationCV.metrics(predictions, screen: screen)
            log(String(format: "candidate %-10@ rms %7.2f px  median %7.2f px  p95 %7.2f px",
                       kind.shortName as NSString, metrics.rms, metrics.median, metrics.p95))
            guard metrics.rms.isFinite else { continue }
            guard var fullFit = MapFitter.fit(kind: kind, samples: usable,
                                              maxCenters: options.maxCenters,
                                              referenceScale: referenceScale) else { continue }
            fullFit.cvRMS = metrics.rms
            candidates.append((fullFit, metrics))
        }
        guard let best = candidates.min(by: { $0.metrics.rms < $1.metrics.rms }) else { return nil }
        let bestRMS = best.metrics.rms

        // 2. Blend weights: sharply favour the winner, drop models that are clearly worse.
        var weights: [Double] = []
        for candidate in candidates {
            let ratio = bestRMS > 1e-6 ? candidate.metrics.rms / bestRMS : 1
            let weight = ratio <= 2.5 ? pow(1.0 / max(ratio, 1e-6), 4.0) : 0
            weights.append(weight)
        }
        let weightSum = weights.reduce(0, +)
        if weightSum > 0 {
            weights = weights.map { $0 / weightSum }
        } else {
            weights = candidates.map { _ in 1.0 / Double(candidates.count) }
        }
        for i in candidates.indices {
            candidates[i].map.weight = weights[i]
        }
        let blendInputs = candidates.map { ($0.map, $0.map.weight) }

        // 3. Cross-validate the blend.
        let blendPredictions = CalibrationCV.run(samples: usable, folds: folds) { training in
            let fitted = candidates.compactMap { candidate -> (FittedMap, Double)? in
                guard let refit = MapFitter.fit(kind: candidate.map.kind, samples: training,
                                                maxCenters: options.maxCenters,
                                                referenceScale: referenceScale) else { return nil }
                return (refit, candidate.map.weight)
            }
            return { sample in
                PointerModel.blend(fitted, at: sample.source, pose: sample.pose) ?? sample.source
            }
        }
        var afterMetrics = CalibrationCV.metrics(blendPredictions, screen: screen)
        if !afterMetrics.rms.isFinite { afterMetrics = best.metrics }
        var finalMetrics = afterMetrics
        var finalPredictions = blendPredictions
        log(String(format: "stage 1 blend rms %7.2f px  median %7.2f px", afterMetrics.rms, afterMetrics.median))

        // 3b. Second stage: fit a spline on the residual of the blend, in output space. Composing
        // stages recovers both the depth stretch (stage 1) and the smooth lens/palm curvature
        // (stage 2); averaging two maps would only recover half of each.
        var residualStage: FittedMap?
        if usable.count >= 40 {
            let stage2Predictions = CalibrationCV.run(samples: usable, folds: folds) { training in
                let fitted = candidates.compactMap { candidate -> (FittedMap, Double)? in
                    guard let refit = MapFitter.fit(kind: candidate.map.kind, samples: training,
                                                    maxCenters: options.maxCenters,
                                                    referenceScale: referenceScale) else { return nil }
                    return (refit, candidate.map.weight)
                }
                let basePredict: (CalibrationSample) -> Vec2 = { sample in
                    PointerModel.blend(fitted, at: sample.source, pose: sample.pose) ?? sample.source
                }
                guard let corrective = residualSamples(training, predict: basePredict),
                      let spline = MapFitter.fit(kind: .poly2, samples: corrective,
                                                 maxCenters: options.maxCenters) else {
                    return basePredict
                }
                return { sample in
                    let base = basePredict(sample)
                    return spline.raw(base, pose: sample.pose) ?? base
                }
            }
            let stage2Metrics = CalibrationCV.metrics(stage2Predictions, screen: screen)
            log(String(format: "stage 2 spline rms %7.2f px  median %7.2f px", stage2Metrics.rms, stage2Metrics.median))
            if stage2Metrics.rms.isFinite, stage2Metrics.rms < afterMetrics.rms * 0.98 {
                let basePredict: (CalibrationSample) -> Vec2 = { sample in
                    PointerModel.blend(blendInputs, at: sample.source, pose: sample.pose) ?? sample.source
                }
                if let corrective = residualSamples(usable, predict: basePredict) {
                    residualStage = MapFitter.fit(kind: .poly2, samples: corrective,
                                                  maxCenters: options.maxCenters)
                }
                if residualStage != nil {
                    finalMetrics = stage2Metrics
                    finalPredictions = stage2Predictions
                }
            }
        }

        // 3c. Local residual table. Global stages smooth over the curvature between calibration
        // dots; a per-query local quadratic follows it, which is where most of the error in real
        // use sits (nobody measures accuracy at the dots themselves).
        var localStage: LocalResidual?
        if usable.count >= 40 {
            let localPredictions = CalibrationCV.run(samples: usable, folds: folds) { training in
                let fitted = candidates.compactMap { candidate -> (FittedMap, Double)? in
                    guard let refit = MapFitter.fit(kind: candidate.map.kind, samples: training,
                                                    maxCenters: options.maxCenters,
                                                    referenceScale: referenceScale) else { return nil }
                    return (refit, candidate.map.weight)
                }
                let basePredict: (CalibrationSample) -> Vec2 = { sample in
                    PointerModel.blend(fitted, at: sample.source, pose: sample.pose) ?? sample.source
                }
                var spline: FittedMap?
                if residualStage != nil, let corrective = residualSamples(training, predict: basePredict) {
                    spline = MapFitter.fit(kind: .poly2, samples: corrective,
                                           maxCenters: options.maxCenters)
                }
                let mapper: (Vec2, PointerPose) -> Vec2 = { point, pose in
                    let base = PointerModel.blend(fitted, at: point, pose: pose) ?? point
                    if let spline, let corrected = spline.raw(base, pose: pose) { return corrected }
                    return base
                }
                var pairs: [(mapped: Vec2, target: Vec2)] = []
                pairs.reserveCapacity(training.count)
                for sample in training {
                    let mapped = mapper(sample.source, sample.pose)
                    guard mapped.isValid else { continue }
                    pairs.append((mapped, sample.target))
                }
                guard let local = LocalResidual(predictions: pairs) else {
                    return { sample in mapper(sample.source, sample.pose) }
                }
                return { sample in
                    let base = mapper(sample.source, sample.pose)
                    if let correction = local.residual(at: base) { return base + correction }
                    return base
                }
            }
            let localMetrics = CalibrationCV.metrics(localPredictions, screen: screen)
            log(String(format: "local residual rms %7.2f px  median %7.2f px",
                       localMetrics.rms, localMetrics.median))
            if localMetrics.rms.isFinite, localMetrics.rms < finalMetrics.rms * 0.995 {
                let mapper: (Vec2, PointerPose) -> Vec2 = { point, pose in
                    let base = PointerModel.blend(blendInputs, at: point, pose: pose) ?? point
                    if let residualStage, let corrected = residualStage.raw(base, pose: pose) {
                        return corrected
                    }
                    return base
                }
                var pairs: [(mapped: Vec2, target: Vec2)] = []
                pairs.reserveCapacity(usable.count)
                for sample in usable {
                    let mapped = mapper(sample.source, sample.pose)
                    guard mapped.isValid else { continue }
                    pairs.append((mapped, sample.target))
                }
                localStage = LocalResidual(predictions: pairs)
                if localStage != nil {
                    finalMetrics = localMetrics
                    finalPredictions = localPredictions
                }
            }
        }

        // 4. Neural residual net — only if cross-validation says it actually improves things, and
        // never alongside the local stage: both learn the same residual and would double-correct.
        var finalNet: MLP?
        var finalScaler: StandardScaler?
        var finalTrust: TrustGate?
        var netReport: TrainingReport?

        if options.useResidualNet, localStage == nil, usable.count >= 40, groupCount >= 6 {
            let netPredictions = CalibrationCV.run(samples: usable, folds: folds) { training in
                let fitted = candidates.compactMap { candidate -> (FittedMap, Double)? in
                    guard let refit = MapFitter.fit(kind: candidate.map.kind, samples: training,
                                                    maxCenters: options.maxCenters,
                                                    referenceScale: referenceScale) else { return nil }
                    return (refit, candidate.map.weight)
                }
                var spline: FittedMap?
                let fitBase: (CalibrationSample) -> Vec2 = { sample in
                    PointerModel.blend(fitted, at: sample.source, pose: sample.pose) ?? sample.source
                }
                if residualStage != nil,
                   let corrective = residualSamples(training, predict: fitBase) {
                    spline = MapFitter.fit(kind: .poly2, samples: corrective,
                                           maxCenters: options.maxCenters)
                }
                let mapper: (Vec2, PointerPose) -> Vec2 = { point, pose in
                    let base = PointerModel.blend(fitted, at: point, pose: pose) ?? point
                    if let spline, let corrected = spline.raw(base, pose: pose) { return corrected }
                    return base
                }
                guard let bundle = ResidualNet.fit(samples: training, mapper: mapper,
                                                   referenceScale: referenceScale, epochs: 550) else {
                    return { sample in mapper(sample.source, sample.pose) }
                }
                return { sample in
                    let base = mapper(sample.source, sample.pose)
                    let scaled = bundle.scaler.transform(
                        PointerFeatures.vector(mapped: base, pose: sample.pose, referenceScale: referenceScale)
                    )
                    let confidence = bundle.trust.weight(for: scaled)
                    let residual = bundle.mlp.predict(scaled)
                    guard confidence > 0, residual.count >= 2,
                          residual[0].isFinite, residual[1].isFinite else { return base }
                    return base + Vec2(residual[0], residual[1]) * confidence
                }
            }
            let netMetrics = CalibrationCV.metrics(netPredictions, screen: screen)
            log(String(format: "residual net rms %7.2f px  median %7.2f px", netMetrics.rms, netMetrics.median))
            if netMetrics.rms.isFinite, netMetrics.rms < finalMetrics.rms * 0.97 {
                finalMetrics = netMetrics
                finalPredictions = netPredictions
                let mapper: (Vec2, PointerPose) -> Vec2 = { point, pose in
                    let base = PointerModel.blend(blendInputs, at: point, pose: pose) ?? point
                    if let residualStage, let corrected = residualStage.raw(base, pose: pose) {
                        return corrected
                    }
                    return base
                }
                if let bundle = ResidualNet.fit(samples: usable, mapper: mapper,
                                                referenceScale: referenceScale, epochs: 800) {
                    finalNet = bundle.mlp
                    finalScaler = bundle.scaler
                    finalTrust = bundle.trust
                    netReport = bundle.report
                }
            }
        }

        // 5. Gaze model, fitted from the same targets (you were looking at them).
        let gazePairs = usable.compactMap { sample -> (features: GazeFeatures, target: Vec2)? in
            guard let gaze = sample.gaze else { return nil }
            return (gaze, sample.target)
        }
        let gaze = GazeModel.fit(gazePairs, screen: screen)

        // 6. Assemble.
        var stats = CalibrationStats()
        stats.samples = usable.count
        stats.targets = groupCount
        stats.rmsBefore = bestRMS
        stats.medianBefore = best.metrics.median
        stats.p95Before = best.metrics.p95
        stats.rmsAfter = finalMetrics.rms
        stats.medianAfter = finalMetrics.median
        stats.p95After = finalMetrics.p95
        stats.screenWidth = screen.width
        stats.screenHeight = screen.height
        stats.chosen = candidates
            .filter { $0.map.weight > 0.02 }
            .map { String(format: "%@ %.0f%%", $0.map.kind.shortName, $0.map.weight * 100) }
        let detailed = detailedStats(baseline: blendPredictions, final: finalPredictions, screen: screen)
        stats.cells = detailed.cells
        stats.errorsBefore = detailed.before
        stats.errorsAfter = detailed.after
        stats.residualSplineIncluded = residualStage != nil
        stats.residualNetIncluded = finalNet != nil
        stats.residualNetValidationLoss = netReport?.validationLoss ?? 0
        stats.gazeRMS = gaze?.rmsPixels
        stats.trainedAt = Date()

        return PointerModel(screen: screen,
                            maps: candidates.map { $0.map },
                            residualStage: residualStage,
                            localResidual: localStage,
                            mlp: finalNet,
                            scaler: finalScaler,
                            trust: finalTrust,
                            referenceScale: referenceScale,
                            gaze: gaze,
                            stats: stats)
    }

    private static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        return sorted.count % 2 == 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
    }

    /// Per-cell accuracy and signed bias, plus the raw error lists, for the accuracy chart.
    private static func detailedStats(baseline: [PredictionError],
                                      final: [PredictionError],
                                      screen: RectD) -> (cells: [CellError], before: [Double], after: [Double]) {
        func distance(_ prediction: Vec2, _ target: Vec2) -> Double {
            let dx = (prediction.x - target.x) * screen.width
            let dy = (prediction.y - target.y) * screen.height
            return (dx * dx + dy * dy).squareRoot()
        }

        let before = baseline.map { distance($0.predicted, $0.target) }
        let after = final.map { distance($0.predicted, $0.target) }

        var beforeByGroup: [Int: [Double]] = [:]
        for prediction in baseline {
            beforeByGroup[prediction.group, default: []].append(distance(prediction.predicted, prediction.target))
        }

        var cells: [CellError] = []
        var afterByGroup: [Int: [Double]] = [:]
        var targetByGroup: [Int: Vec2] = [:]
        var biasByGroup: [Int: Vec2] = [:]
        var countByGroup: [Int: Int] = [:]

        for prediction in final {
            let error = distance(prediction.predicted, prediction.target)
            afterByGroup[prediction.group, default: []].append(error)
            targetByGroup[prediction.group] = prediction.target
            let bias = Vec2((prediction.predicted.x - prediction.target.x) * screen.width,
                            (prediction.predicted.y - prediction.target.y) * screen.height)
            biasByGroup[prediction.group, default: .zero] += bias
            countByGroup[prediction.group, default: 0] += 1
        }

        for (group, errors) in afterByGroup {
            guard let target = targetByGroup[group], let count = countByGroup[group], count > 0 else { continue }
            let biasSum = biasByGroup[group] ?? .zero
            cells.append(CellError(target: target,
                                   count: count,
                                   medianBefore: median(beforeByGroup[group] ?? []),
                                   medianAfter: median(errors),
                                   bias: Vec2(biasSum.x / Double(count), biasSum.y / Double(count))))
        }
        cells.sort { $0.target.y == $1.target.y ? $0.target.x < $1.target.x : $0.target.y < $1.target.y }

        return (cells, before, after)
    }

    /// Per-stage diagnostics, enabled with `AEROMOUSE_DEBUG=1`.
    private static func log(_ message: @autoclosure () -> String) {
        guard ProcessInfo.processInfo.environment["AEROMOUSE_DEBUG"] != nil else { return }
        FileHandle.standardError.write(Data((message() + "\n").utf8))
    }

    /// Re-keys samples so a stage can be fitted in output space: `predict(sample)` becomes the
    /// new source and the original target stays the target.
    private static func residualSamples(_ samples: [CalibrationSample],
                                        predict: (CalibrationSample) -> Vec2) -> [CalibrationSample]? {
        var out: [CalibrationSample] = []
        out.reserveCapacity(samples.count)
        for sample in samples {
            let base = predict(sample)
            guard base.isValid else { continue }
            out.append(CalibrationSample(source: base,
                                         target: sample.target,
                                         weight: sample.weight,
                                         group: sample.group,
                                         pose: sample.pose,
                                         gaze: sample.gaze))
        }
        return out.isEmpty ? nil : out
    }

    private static func medianScale(_ samples: [CalibrationSample]) -> Double {
        let values = samples.map { $0.pose.handScale }.filter { $0.isFinite && $0 > 1e-6 }.sorted()
        guard !values.isEmpty else { return 0.15 }
        return values[values.count / 2]
    }
}
