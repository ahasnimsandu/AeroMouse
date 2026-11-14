import Foundation
import AVFoundation
import CoreMedia
import CoreVideo

enum CameraError: LocalizedError {
    case notFound
    case cannotAddInput
    case cannotAddOutput

    var errorDescription: String? {
        switch self {
        case .notFound: return "No camera found."
        case .cannotAddInput: return "The camera could not be attached."
        case .cannotAddOutput: return "The camera output could not be attached."
        }
    }
}

/// Owns the capture session. Chooses the highest frame rate the camera offers (up to 60 fps) at
/// the smallest usable resolution, because Vision latency scales with pixels and latency is what
/// makes a hand mouse feel wrong.
final class CameraCapture: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let processingQueue = DispatchQueue(label: "com.aeromouse.camera", qos: .userInteractive)

    private(set) var framesPerSecond: Double = 0
    private(set) var activeFrameRate: Double = 0
    private(set) var activeWidth: Int = 0
    private(set) var activeHeight: Int = 0
    private var configured = false
    private var lastFrameTime: Double = 0
    private var fpsEstimate: Double = 0
    private var device: AVCaptureDevice?

    var preferredFPS: Double = 60
    /// Landmark precision is set by how many pixels the hand covers, so a larger sensor mode is
    /// worth a little latency. Turned off for a constant-60-fps trade.
    var preferHighResolution: Bool = true
    var onFrame: ((CVPixelBuffer, Double) -> Void)?

    func start() -> String? {
        if !configured {
            do {
                try configure()
            } catch {
                return error.localizedDescription
            }
        }
        if !session.isRunning {
            processingQueue.async { [session] in
                session.startRunning()
            }
        }
        return nil
    }

    func stop() {
        processingQueue.async { [session] in
            if session.isRunning { session.stopRunning() }
        }
    }

    private func configure() throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        // No session preset: any concrete preset (the old code set .vga640x480) makes AVFoundation
        // ignore the device format chosen below, so the camera was pinned at 640×480 no matter what
        // configureFormat decided. On macOS the session follows the input device's activeFormat.

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
            ?? AVCaptureDevice.default(for: .video) else {
            throw CameraError.notFound
        }

        self.device = camera
        let input = try AVCaptureDeviceInput(device: camera)
        guard session.canAddInput(input) else { throw CameraError.cannotAddInput }
        session.addInput(input)

        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: processingQueue)
        guard session.canAddOutput(output) else { throw CameraError.cannotAddOutput }
        session.addOutput(output)

        configureFormat(camera)
        configured = true
    }

    /// Ranks a sensor mode. Resolution dominates because a hand that spans twice as many pixels
    /// gives landmarks with half the positional error, and that error is multiplied again on the
    /// way to the screen — 720p roughly halves pointer noise compared with 640×480.
    static func score(width: Int, height: Int, maxRate: Double, preferHighResolution: Bool) -> Double {
        guard width >= 640, height >= 480, maxRate >= 24 else { return -.infinity }
        let resolution: Double
        switch width {
        case 1280...1600: resolution = 3.0     // the sweet spot: sharp and still fast
        case 1601...: resolution = 2.5          // sharper, but Vision latency grows
        case 960..<1280: resolution = 1.6
        default: resolution = 1.0               // 640×480 and other small modes
        }
        let rate: Double = maxRate >= 55 ? 2 : (maxRate >= 29 ? 1 : 0)
        if preferHighResolution {
            return resolution * 10 + rate
        }
        // Fast mode prefers the small sensor mode, where frame rate is cheap to keep high.
        let small: Double = width <= 800 ? 3.0 : (width <= 1300 ? 1.5 : 1.0)
        return small * 10 + rate * 1.5
    }

    private func configureFormat(_ camera: AVCaptureDevice) {
        var best: AVCaptureDevice.Format?
        var bestScore = -Double.infinity

        for format in camera.formats {
            let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            let maxRate = format.videoSupportedFrameRateRanges.map { $0.maxFrameRate }.max() ?? 0
            let score = Self.score(width: Int(dimensions.width), height: Int(dimensions.height),
                                   maxRate: maxRate, preferHighResolution: preferHighResolution)
            if score > bestScore {
                bestScore = score
                best = format
            }
        }

        guard let chosen = best else { return }
        let chosenDimensions = CMVideoFormatDescriptionGetDimensions(chosen.formatDescription)
        do {
            try camera.lockForConfiguration()
            camera.activeFormat = chosen
            activeWidth = Int(chosenDimensions.width)
            activeHeight = Int(chosenDimensions.height)
            if let range = chosen.videoSupportedFrameRateRanges.first {
                let maxRate = chosen.videoSupportedFrameRateRanges.map { $0.maxFrameRate }.max() ?? 30
                let desired = min(preferredFPS, maxRate)
                let ticks = max(Int(desired.rounded()), 1)
                let duration = CMTime(value: 1, timescale: CMTimeScale(ticks))
                let clamped = CMTimeMaximum(duration, range.minFrameDuration)
                camera.activeVideoMinFrameDuration = clamped
                camera.activeVideoMaxFrameDuration = clamped
                activeFrameRate = desired
            }
            camera.unlockForConfiguration()
        } catch {
            // Keep the default format; tracking still works at whatever the camera provides.
        }
    }

    /// Changes the capture rate while running. Dropping to 30 fps in dim conditions lets the camera
    /// integrate light for twice as long, which cuts sensor noise at the source — far more effective
    /// than any amount of post-processing.
    func setFrameRate(_ fps: Double) {
        processingQueue.async { [weak self] in
            guard let self, let camera = self.device else { return }
            guard let range = camera.activeFormat.videoSupportedFrameRateRanges.first else { return }
            let supported = camera.activeFormat.videoSupportedFrameRateRanges
            let maximum = supported.map { $0.maxFrameRate }.max() ?? 30
            let desired = min(max(fps, 15), maximum)
            guard abs(desired - self.activeFrameRate) > 0.5 else { return }

            let duration = CMTimeMaximum(CMTime(value: 1, timescale: CMTimeScale(max(Int(desired.rounded()), 1))),
                                         range.minFrameDuration)
            do {
                try camera.lockForConfiguration()
                camera.activeVideoMinFrameDuration = duration
                camera.activeVideoMaxFrameDuration = duration
                camera.unlockForConfiguration()
                self.activeFrameRate = desired
            } catch {
                // Keep the existing rate; tracking continues either way.
            }
        }
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let seconds = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        let timestamp = seconds.isFinite ? seconds : CACurrentMediaTime()

        if lastFrameTime > 0, timestamp > lastFrameTime {
            let instant = 1.0 / max(timestamp - lastFrameTime, 1e-4)
            fpsEstimate = fpsEstimate == 0 ? instant : fpsEstimate * 0.9 + instant * 0.1
            framesPerSecond = fpsEstimate
        }
        lastFrameTime = timestamp

        onFrame?(pixelBuffer, timestamp)
    }
}
