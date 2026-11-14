import Foundation

// MARK: - Deterministic RNG

struct SplitMix64 {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed &* 0x9E3779B97F4A7C15 &+ 0x123456789ABCDEF
    }

    mutating func next() -> UInt64 {
        state = state &+ 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    mutating func nextDouble() -> Double {
        Double(next() >> 11) / 9007199254740992.0
    }

    mutating func nextInt(_ upperBound: Int) -> Int {
        guard upperBound > 1 else { return 0 }
        return Int(next() % UInt64(upperBound))
    }
}

// MARK: - Feature scaling

struct StandardScaler: Codable, Equatable {
    var mean: [Double]
    var std: [Double]

    static func fit(_ rows: [[Double]]) -> StandardScaler {
        guard let first = rows.first, !first.isEmpty else {
            return StandardScaler(mean: [], std: [])
        }
        let d = first.count
        var mean = [Double](repeating: 0, count: d)
        for row in rows where row.count == d {
            for i in 0..<d { mean[i] += row[i] }
        }
        let n = Double(max(rows.count, 1))
        for i in 0..<d { mean[i] /= n }

        var variance = [Double](repeating: 0, count: d)
        for row in rows where row.count == d {
            for i in 0..<d {
                let dv = row[i] - mean[i]
                variance[i] += dv * dv
            }
        }
        var std = [Double](repeating: 1, count: d)
        for i in 0..<d {
            std[i] = max((variance[i] / Double(max(rows.count - 1, 1))).squareRoot(), 1e-6)
        }
        return StandardScaler(mean: mean, std: std)
    }

    func transform(_ row: [Double]) -> [Double] {
        guard mean.count == row.count else { return row }
        return (0..<row.count).map { (row[$0] - mean[$0]) / std[$0] }
    }
}

// MARK: - Multilayer perceptron (trained from scratch, no third-party runtime)

struct MLP: Codable, Equatable {
    var sizes: [Int]
    var weights: [[Double]]   // layer l: [out][in] flattened, row-major
    var biases: [[Double]]

    init(sizes: [Int], seed: UInt64 = 1) {
        self.sizes = sizes
        weights = []
        biases = []
        var rng = SplitMix64(seed: seed)
        guard sizes.count >= 2 else { return }
        for l in 0..<(sizes.count - 1) {
            let fanIn = sizes[l]
            let fanOut = sizes[l + 1]
            let limit = (6.0 / Double(fanIn + fanOut)).squareRoot()
            var w = [Double](repeating: 0, count: fanIn * fanOut)
            for i in 0..<w.count {
                w[i] = (rng.nextDouble() * 2 - 1) * limit
            }
            weights.append(w)
            biases.append([Double](repeating: 0, count: fanOut))
        }
    }

    /// Returns the output plus the full list of layer activations (input first).
    func forward(_ input: [Double]) -> (output: [Double], activations: [[Double]]) {
        guard let inputSize = sizes.first else { return (input, [input]) }
        var current = input
        if current.count != inputSize {
            var padded = [Double](repeating: 0, count: inputSize)
            for i in 0..<min(inputSize, current.count) { padded[i] = current[i] }
            current = padded
        }
        var activations: [[Double]] = [current]
        let lastLayer = weights.count - 1
        for l in 0..<weights.count {
            let inN = sizes[l]
            let outN = sizes[l + 1]
            let w = weights[l]
            var z = biases[l]
            for o in 0..<outN {
                var sum = z[o]
                let base = o * inN
                for i in 0..<inN { sum += w[base + i] * current[i] }
                z[o] = sum
            }
            if l < lastLayer {
                for o in 0..<outN { z[o] = tanh(z[o]) }
            }
            activations.append(z)
            current = z
        }
        return (current, activations)
    }

    func predict(_ input: [Double]) -> [Double] {
        forward(input).output
    }

    var parameterCount: Int {
        weights.reduce(0) { $0 + $1.count } + biases.reduce(0) { $0 + $1.count }
    }
}

// MARK: - Trainer (Adam + early stopping on a held-out split)

struct TrainingConfig {
    var epochs: Int = 600
    var batchSize: Int = 32
    var learningRate: Double = 0.01
    var weightDecay: Double = 1e-5
    var validationFraction: Double = 0.25
    var patience: Int = 150
    var seed: UInt64 = 7
}

struct TrainingReport {
    var trainLoss: Double = .infinity
    var validationLoss: Double = .infinity
    var bestEpoch: Int = 0
    var epochsRun: Int = 0
}

private struct AdamState {
    var mw: [[Double]]
    var vw: [[Double]]
    var mb: [[Double]]
    var vb: [[Double]]
    var t: Int = 0

    init(model: MLP) {
        mw = model.weights.map { [Double](repeating: 0, count: $0.count) }
        vw = mw
        mb = model.biases.map { [Double](repeating: 0, count: $0.count) }
        vb = mb
    }
}

enum MLPTrainer {
    static func train(_ model: inout MLP,
                      inputs: [[Double]],
                      outputs: [[Double]],
                      config: TrainingConfig) -> TrainingReport {
        guard !inputs.isEmpty, inputs.count == outputs.count,
              let inDim = inputs.first?.count, let outDim = outputs.first?.count,
              model.sizes.first == inDim, model.sizes.last == outDim else {
            return TrainingReport()
        }

        var rng = SplitMix64(seed: config.seed)
        var order = Array(0..<inputs.count)
        shuffle(&order, &rng)

        let validationCount = inputs.count >= 24 ? max(1, Int(Double(inputs.count) * config.validationFraction)) : 0
        let validation = Array(order.prefix(validationCount))
        let training = Array(order.dropFirst(validationCount))
        guard !training.isEmpty else { return TrainingReport() }

        var best = model
        var bestLoss = Double.infinity
        var bestEpoch = 0
        var lastTrainLoss = Double.infinity
        var state = AdamState(model: model)
        var sinceImprovement = 0
        var epochsRun = 0

        for epoch in 0..<config.epochs {
            epochsRun = epoch + 1
            var shuffled = training
            shuffle(&shuffled, &rng)

            var epochLoss = 0.0
            var batches = 0
            var start = 0

            while start < shuffled.count {
                let end = min(start + config.batchSize, shuffled.count)
                let batch = Array(shuffled[start..<end])
                var gw = model.weights.map { [Double](repeating: 0, count: $0.count) }
                var gb = model.biases.map { [Double](repeating: 0, count: $0.count) }
                var batchLoss = 0.0

                for index in batch {
                    let (prediction, activations) = model.forward(inputs[index])
                    var dOut = [Double](repeating: 0, count: outDim)
                    for o in 0..<outDim {
                        let diff = prediction[o] - outputs[index][o]
                        batchLoss += diff * diff
                        dOut[o] = 2 * diff
                    }
                    accumulate(model: model, activations: activations, dOut: dOut, gw: &gw, gb: &gb)
                }

                let n = Double(batch.count)
                epochLoss += batchLoss / n
                batches += 1
                for l in 0..<gw.count {
                    for i in 0..<gw[l].count { gw[l][i] /= n }
                    for i in 0..<gb[l].count { gb[l][i] /= n }
                }
                adamStep(model: &model, gw: gw, gb: gb, state: &state, config: config)
                start = end
            }

            lastTrainLoss = batches > 0 ? epochLoss / Double(batches) : .infinity

            if validation.isEmpty {
                best = model
                bestLoss = lastTrainLoss
                bestEpoch = epoch + 1
            } else {
                var validationLoss = 0.0
                for index in validation {
                    let prediction = model.predict(inputs[index])
                    for o in 0..<outDim {
                        let diff = prediction[o] - outputs[index][o]
                        validationLoss += diff * diff
                    }
                }
                validationLoss /= Double(validation.count)
                if validationLoss < bestLoss - 1e-10 {
                    bestLoss = validationLoss
                    best = model
                    bestEpoch = epoch + 1
                    sinceImprovement = 0
                } else {
                    sinceImprovement += 1
                    if sinceImprovement > config.patience { break }
                }
            }
        }

        if validation.isEmpty, bestEpoch == 0 {
            best = model
            bestLoss = lastTrainLoss
            bestEpoch = epochsRun
        }
        model = best
        return TrainingReport(trainLoss: lastTrainLoss,
                              validationLoss: bestLoss,
                              bestEpoch: bestEpoch,
                              epochsRun: epochsRun)
    }

    private static func shuffle(_ array: inout [Int], _ rng: inout SplitMix64) {
        guard array.count > 1 else { return }
        for i in stride(from: array.count - 1, to: 0, by: -1) {
            let j = rng.nextInt(i + 1)
            array.swapAt(i, j)
        }
    }

    private static func accumulate(model: MLP,
                                   activations: [[Double]],
                                   dOut: [Double],
                                   gw: inout [[Double]],
                                   gb: inout [[Double]]) {
        let layers = model.weights.count
        guard layers > 0 else { return }
        var delta = dOut

        for l in stride(from: layers - 1, through: 0, by: -1) {
            let inN = model.sizes[l]
            let outN = model.sizes[l + 1]
            var dz = delta
            if l < layers - 1 {
                for o in 0..<outN {
                    let a = activations[l + 1][o]
                    dz[o] = delta[o] * (1 - a * a)
                }
            }
            let input = activations[l]
            for o in 0..<outN {
                let base = o * inN
                for i in 0..<inN { gw[l][base + i] += dz[o] * input[i] }
                gb[l][o] += dz[o]
            }
            if l > 0 {
                var previous = [Double](repeating: 0, count: inN)
                for o in 0..<outN {
                    let base = o * inN
                    for i in 0..<inN { previous[i] += dz[o] * model.weights[l][base + i] }
                }
                delta = previous
            }
        }
    }

    private static func adamStep(model: inout MLP,
                                 gw: [[Double]],
                                 gb: [[Double]],
                                 state: inout AdamState,
                                 config: TrainingConfig) {
        state.t += 1
        let beta1 = 0.9
        let beta2 = 0.999
        let eps = 1e-8
        let correction1 = 1 - pow(beta1, Double(state.t))
        let correction2 = 1 - pow(beta2, Double(state.t))

        for l in 0..<model.weights.count {
            for i in 0..<model.weights[l].count {
                let g = gw[l][i] + config.weightDecay * model.weights[l][i]
                state.mw[l][i] = beta1 * state.mw[l][i] + (1 - beta1) * g
                state.vw[l][i] = beta2 * state.vw[l][i] + (1 - beta2) * g * g
                let mHat = state.mw[l][i] / correction1
                let vHat = state.vw[l][i] / correction2
                model.weights[l][i] -= config.learningRate * mHat / (vHat.squareRoot() + eps)
            }
            for i in 0..<model.biases[l].count {
                let g = gb[l][i]
                state.mb[l][i] = beta1 * state.mb[l][i] + (1 - beta1) * g
                state.vb[l][i] = beta2 * state.vb[l][i] + (1 - beta2) * g * g
                let mHat = state.mb[l][i] / correction1
                let vHat = state.vb[l][i] / correction2
                model.biases[l][i] -= config.learningRate * mHat / (vHat.squareRoot() + eps)
            }
        }
    }
}
