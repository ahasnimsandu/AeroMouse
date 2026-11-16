import Foundation
import CoreImage
import CoreImage.CIFilterBuiltins
import CoreVideo
import Metal
import CoreGraphics

/// What to do with a frame, given how bright it is.
struct LowLightDecision: Equatable {
    var enhance: Bool
    var shadowLift: Double      // 0...0.5 tone-curve lift
    var temporalBlend: Double   // weight of the previous frame (0 = none)
    var targetFPS: Double
}

enum LowLightPolicy {
    /// Bright frames are passed straight through. Dark frames get a tone-curve shadow lift, a light
    /// unsharp pass (which restores the edge structure a noisy sensor smears), temporal averaging,
    /// and a lower frame rate so the camera integrates light for longer — the single biggest win in
    /// low light, because it reduces sensor noise at the source.
    static func decision(forLuma luma: Double) -> LowLightDecision {
        if luma >= 0.50 {
            return LowLightDecision(enhance: false, shadowLift: 0, temporalBlend: 0, targetFPS: 60)
        }
        if luma >= 0.28 {
            return LowLightDecision(enhance: true, shadowLift: 0.24, temporalBlend: 0.38, targetFPS: 30)
        }
        return LowLightDecision(enhance: true, shadowLift: 0.46, temporalBlend: 0.62, targetFPS: 30)
    }
}

/// GPU frame conditioning using Core Image (Metal-backed). Everything runs on the GPU; nothing
/// here touches the CPU pixel data except a sparse luminance sample used to pick the mode.
final class FrameEnhancer {
    private let context: CIContext
    private var pool: CVPixelBufferPool?
    private var poolSize: CGSize = .zero
    private var previous: CIImage?

    private(set) var lastLuma: Double = 0.5

    init() {
        // Metal-backed context: Core Image dispatches every filter below to the GPU.
        context = CIContext(options: [
            .useSoftwareRenderer: false,
            .cacheIntermediates: true,
            .priorityRequestLow: false,
        ])
    }

    /// Sparse 16-pixel-stride luma sample. Cheap enough at 60 fps and needs no GPU readback.
    func measureLuma(_ pixelBuffer: CVPixelBuffer) -> Double {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return 0.5 }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard width > 0, height > 0, bytesPerRow > 0 else { return 0.5 }

        let bytes = base.assumingMemoryBound(to: UInt8.self)
        let step = 16
        var total = 0.0
        var samples = 0

        var y = 0
        while y < height {
            let row = bytes + y * bytesPerRow
            var x = 0
            while x < width {
                let pixel = row + x * 4
                let blue = Double(pixel[0])
                let green = Double(pixel[1])
                let red = Double(pixel[2])
                total += (0.114 * blue + 0.587 * green + 0.299 * red) / 255.0
                samples += 1
                x += step
            }
            y += step
        }

        guard samples > 0 else { return 0.5 }
        let luma = total / Double(samples)
        lastLuma = luma
        return luma
    }

    func enhance(_ pixelBuffer: CVPixelBuffer, decision: LowLightDecision) -> CVPixelBuffer? {
        guard decision.enhance else {
            previous = CIImage(cvPixelBuffer: pixelBuffer)
            return pixelBuffer
        }

        var image = CIImage(cvPixelBuffer: pixelBuffer)

        let lift = decision.shadowLift
        if lift > 0 {
            let curve = CIFilter.toneCurve()
            curve.inputImage = image
            curve.point0 = CGPoint(x: 0, y: 0.01 + lift)
            curve.point1 = CGPoint(x: 0.25, y: 0.25 + lift * 0.9)
            curve.point2 = CGPoint(x: 0.50, y: 0.50 + lift * 0.6)
            curve.point3 = CGPoint(x: 0.75, y: 0.75 + lift * 0.25)
            curve.point4 = CGPoint(x: 1.0, y: 1.0)
            image = curve.outputImage ?? image

            // Lift exposure and crush sensor noise before sharpening, otherwise the unsharp pass
            // amplifies exactly the grain that makes landmarks jump in the dark.
            let exposure = CIFilter.exposureAdjust()
            exposure.inputImage = image
            exposure.ev = Float(lift * 1.5)
            image = exposure.outputImage ?? image

            let noise = CIFilter.noiseReduction()
            noise.inputImage = image
            noise.noiseLevel = 0.02
            noise.sharpness = 0.35
            image = noise.outputImage ?? image

            let sharp = CIFilter.unsharpMask()
            sharp.inputImage = image
            sharp.radius = 1.2
            sharp.intensity = 0.40
            image = sharp.outputImage ?? image
        }

        if let previous, decision.temporalBlend > 0,
           let blend = CIFilter(name: "CIDissolveTransition") {
            // Kernel-level filter: the typed builtins do not expose this transition's time key.
            blend.setValue(previous, forKey: kCIInputImageKey)
            blend.setValue(image, forKey: "inputTargetImage")
            blend.setValue(1 - decision.temporalBlend, forKey: kCIInputTimeKey)
            image = blend.outputImage ?? image
        }
        previous = image

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard let pool = ensurePool(width: width, height: height) else { return nil }

        var output: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &output) == kCVReturnSuccess,
              let target = output else { return nil }

        context.render(image,
                       to: target,
                       bounds: CGRect(x: 0, y: 0, width: width, height: height),
                       colorSpace: CGColorSpaceCreateDeviceRGB())
        return target
    }

    func reset() {
        previous = nil
    }

    private func ensurePool(width: Int, height: Int) -> CVPixelBufferPool? {
        let size = CGSize(width: width, height: height)
        if let pool, poolSize == size { return pool }

        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        let poolAttributes: [String: Any] = [
            kCVPixelBufferPoolMinimumBufferCountKey as String: 4,
        ]
        var created: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(nil, poolAttributes as CFDictionary,
                                      attributes as CFDictionary, &created) == kCVReturnSuccess else {
            return nil
        }
        pool = created
        poolSize = size
        return created
    }
}
