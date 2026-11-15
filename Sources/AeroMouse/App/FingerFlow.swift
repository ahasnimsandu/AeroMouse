import Foundation
import CoreVideo

/// Sub-pixel refinement of the pointer anchor by block matching on the video luma.
///
/// Vision returns an absolute landmark every frame, but its positional noise is a pixel or two in
/// image space — which the screen mapping multiplies into several screen pixels. The *motion*
/// between two consecutive frames can be measured far more precisely than that, so this tracker
/// matches a small patch from the previous frame against the current one, refines the match to a
/// fraction of a pixel, and carries the patch forward. Vision stays the absolute reference: the
/// landmark corrects any drift of the patch, and the patch removes the landmark's frame-to-frame
/// noise. The result is a tiny Kalman filter whose process model is measured image evidence rather
/// than an assumption about velocity.
///
/// Every safety rail falls back to "use the landmark": weak texture, a failed match, a match on the
/// search border, a large disagreement between flow and landmark, or a format it does not recognise.
final class FingerFlow {
    struct Config {
        var patchSize = 34          // px, even is fine: the patch is centred with integer division
        var searchRadius = 8        // px
        var minContrast = 5.0       // standard deviation in gray levels
        var maxMatchCost = 900.0    // mean squared difference per pixel
        var gain = 0.6              // share of the landmark correction applied per frame
        var rejectRadius = 0.035    // normalised screen-units; beyond this the landmark wins outright
    }

    private(set) var isLocked = false
    private(set) var lastMatchCost: Double = .infinity
    private(set) var rescueCount = 0

    private var config = Config()
    private var patch: [UInt8] = []
    private var patchCenter = (x: 0, y: 0)
    private var frameSize = (width: 0, height: 0)
    private var velocity = (x: 0.0, y: 0.0)
    private var hasPatch = false
    private var frameCounter = 0

    func configure(_ config: Config) { self.config = config }

    func reset() {
        patch.removeAll(keepingCapacity: true)
        hasPatch = false
        isLocked = false
        lastMatchCost = .infinity
        velocity = (0, 0)
        rescueCount = 0
        frameCounter = 0
    }

    /// Returns the refined anchor in the caller's aspect-corrected image space, or nil when the
    /// landmark should simply be used as it is.
    func refine(anchor: Vec2, buffer: CVPixelBuffer, time: Double) -> Vec2? {
        guard anchor.isValid else { reset(); return nil }

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA else {
            reset()
            return nil
        }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        guard width > 96, height > 96,
              let base = CVPixelBufferGetBaseAddress(buffer) else { reset(); return nil }
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        let pixels = base.assumingMemoryBound(to: UInt8.self)

        // Only one pixel format is handled, so the size is known: 4 bytes per pixel.
        guard rowBytes >= width * 4 else { reset(); return nil }

        let aspect = Double(width) / Double(height)
        let visionPixel = (x: anchor.x / aspect * Double(width),
                           y: (1 - anchor.y) * Double(height))
        let visionCenter = (x: Int(visionPixel.x.rounded()), y: Int(visionPixel.y.rounded()))

        guard hasPatch, frameSize.width == width, frameSize.height == height else {
            seed(at: visionCenter, pixels: pixels, rowBytes: rowBytes, width: width, height: height)
            return nil
        }

        let predicted = (x: patchCenter.x + Int(velocity.x.rounded()),
                         y: patchCenter.y + Int(velocity.y.rounded()))

        guard let shift = match(pixels: pixels, rowBytes: rowBytes, width: width, height: height,
                               center: predicted) else {
            // Nothing confident this frame: hand authority to the landmark and reseed from it.
            seed(at: visionCenter, pixels: pixels, rowBytes: rowBytes, width: width, height: height)
            rescueCount += 1
            return nil
        }

        let found = (x: predicted.x + Int(shift.x.rounded()), y: predicted.y + Int(shift.y.rounded()))
        let foundSubpixel = (x: Double(found.x) + (shift.x - shift.x.rounded()),
                             y: Double(found.y) + (shift.y - shift.y.rounded()))
        let flowAnchor = Vec2(foundSubpixel.x / Double(width) * aspect,
                              1 - foundSubpixel.y / Double(height))
        guard flowAnchor.isValid else {
            seed(at: visionCenter, pixels: pixels, rowBytes: rowBytes, width: width, height: height)
            return nil
        }

        let innovation = anchor - flowAnchor
        guard innovation.isValid else { return nil }

        // Flow and landmark disagreeing by a lot is not motion — it is a jump, a new finger, or a
        // patch that latched onto the wrong thing. The landmark always wins those.
        if innovation.length > config.rejectRadius {
            seed(at: visionCenter, pixels: pixels, rowBytes: rowBytes, width: width, height: height)
            rescueCount += 1
            return anchor
        }

        let refined = flowAnchor + innovation * config.gain
        velocity = (velocity.x * 0.4 + (foundSubpixel.x - Double(patchCenter.x)) * 0.6,
                    velocity.y * 0.4 + (foundSubpixel.y - Double(patchCenter.y)) * 0.6)
        // Keep the carried patch near the finger: reseed every few frames from the fused estimate so
        // small errors cannot accumulate into a slow drift off the finger.
        frameCounter += 1
        if rescueCount > 0 || frameCounter % 12 == 0 {
            seed(at: (x: Int(refinedPixelX(refined, width: width, aspect: aspect)),
                      y: Int((1 - refined.y) * Double(height))),
                 pixels: pixels, rowBytes: rowBytes, width: width, height: height,
                 resetVelocity: rescueCount > 0)
        } else {
            patchCenter = found
            frameSize = (width, height)
            hasPatch = true
        }
        isLocked = true
        return refined
    }

    private func refinedPixelX(_ point: Vec2, width: Int, aspect: Double) -> Double {
        point.x / aspect * Double(width)
    }

    // MARK: Patch handling

    private func seed(at center: (x: Int, y: Int), pixels: UnsafeMutablePointer<UInt8>,
                      rowBytes: Int, width: Int, height: Int, resetVelocity: Bool = true) {
        guard let values = sample(pixels: pixels, rowBytes: rowBytes, width: width, height: height,
                                  center: center) else {
            hasPatch = false
            isLocked = false
            return
        }
        patch = values
        patchCenter = center
        frameSize = (width, height)
        hasPatch = true
        isLocked = true
        // A periodic reseed is just housekeeping: dropping the motion estimate there would make
        // the tracker lag for a frame every fifth of a second.
        if resetVelocity { velocity = (0, 0) }
    }

    private func sample(pixels: UnsafeMutablePointer<UInt8>, rowBytes: Int,
                        width: Int, height: Int,
                        center: (x: Int, y: Int)) -> [UInt8]? {
        let size = max(config.patchSize, 8)
        let half = size / 2
        guard center.x - half >= 0, center.x + half < width,
              center.y - half >= 0, center.y + half < height else { return nil }

        var values = [UInt8](repeating: 0, count: size * size)
        var sum = 0.0
        var sumSquares = 0.0
        for row in 0..<size {
            let sourceBase = (center.y - half + row) * rowBytes
            for column in 0..<size {
                // BGRA: the green channel is the cheapest good luma approximation.
                let value = pixels[sourceBase + (center.x - half + column) * 4 + 1]
                values[row * size + column] = value
                let d = Double(value)
                sum += d
                sumSquares += d * d
            }
        }
        let count = Double(values.count)
        let variance = sumSquares / count - (sum / count) * (sum / count)
        guard variance > 0, variance.squareRoot() >= config.minContrast else { return nil }
        return values
    }

    /// Integer search around `center`, refined to sub-pixel by a parabola through the minimum.
    /// Returns the offset in pixels from `center`.
    private func match(pixels: UnsafeMutablePointer<UInt8>, rowBytes: Int,
                       width: Int, height: Int, center: (x: Int, y: Int)) -> Vec2? {
        let size = max(config.patchSize, 8)
        let half = size / 2
        let radius = max(config.searchRadius, 1)
        guard patch.count == size * size,
              center.x - half - radius >= 0, center.x + half + radius < width,
              center.y - half - radius >= 0, center.y + half + radius < height else { return nil }

        let span = 2 * radius + 1
        var costs = [Double](repeating: .infinity, count: span * span)
        var bestCost = Double.infinity
        var bestX = 0
        var bestY = 0

        for dy in -radius...radius {
            for dx in -radius...radius {
                var total = 0.0
                for row in 0..<size {
                    let sourceBase = (center.y - half + dy + row) * rowBytes
                    let patchBase = row * size
                    for column in 0..<size {
                        let difference = Double(pixels[sourceBase + (center.x - half + dx + column) * 4 + 1])
                            - Double(patch[patchBase + column])
                        total += difference * difference
                    }
                }
                let cost = total / Double(patch.count)
                costs[(dy + radius) * span + (dx + radius)] = cost
                if cost < bestCost {
                    bestCost = cost
                    bestX = dx
                    bestY = dy
                }
            }
        }

        lastMatchCost = bestCost
        guard bestCost.isFinite, bestCost < config.maxMatchCost else { return nil }
        // A minimum on the border means the true match sits outside the window: not trustworthy.
        guard abs(bestX) < radius, abs(bestY) < radius else { return nil }

        func at(_ dx: Int, _ dy: Int) -> Double { costs[(dy + radius) * span + (dx + radius)] }
        func subpixel(_ minus: Double, _ middle: Double, _ plus: Double) -> Double {
            let denominator = minus - 2 * middle + plus
            guard denominator.isFinite, abs(denominator) > 1e-9 else { return 0 }
            let offset = 0.5 * (minus - plus) / denominator
            return abs(offset) <= 1 ? offset : 0
        }

        let subX = subpixel(at(bestX - 1, bestY), bestCost, at(bestX + 1, bestY))
        let subY = subpixel(at(bestX, bestY - 1), bestCost, at(bestX, bestY + 1))
        return Vec2(Double(bestX) + subX, Double(bestY) + subY)
    }
}
