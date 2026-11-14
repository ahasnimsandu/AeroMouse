import Foundation
import CoreVideo
import CoreMedia
import Vision
import AVFoundation

// MARK: - Snapshot for the UI

struct TrackingSnapshot {
    var isRunning = false
    var hasHand = false
    var handPoints: [VNHumanHandPoseObservation.JointName: Vec2] = [:]
    var anchor: Vec2?
    var cursor: Vec2?
    var pinchIndex: Double = .infinity
    var pinchMiddle: Double = .infinity
    var fist: Double = 0
    var dragging = false
    var frozen = false
    var fps: Double = 0
    var targetFPS: Double = 60
    var visionMilliseconds: Double = 0
    var jitter: Double = 0
    var hasFace = false
    var gaze: Vec2?
    var calibrated = false
    var sampleCount = 0
    var quality: Double = 0
    var luma: Double = 0
    var enhancing = false
    var cameraWidth: Int = 0
    var cameraHeight: Int = 0
    var thumbContact: Double = .infinity
    var gazeAccuracy: Double = 0
    var gazeActive = false
    var gazeOpenness: Double = 0
}

// MARK: - Engine

/// Glue between capture, GPU conditioning, Vision, the calibrated model, gestures and the cursor.
/// Everything here runs on the camera queue, so the per-frame path has no locks except one short
/// settings read.
final class TrackingEngine {
    private let capture = CameraCapture()
    private let vision = VisionPipeline()
    private let cursor = CursorController()
    private let pointer = PointerPipeline()
    private let gestures = GestureRecognizer()
    private let clickLock = ClickLock()
    private let jitterMeter = JitterMeter()
    private let gazeFilter = OneEuroFilter2D()
    private let enhancer = FrameEnhancer()
    private let landmarkSmoother = LandmarkSmoother()

    private let stateLock = NSLock()
    private var settings = AppSettings()
    private var model: PointerModel?
    private var screenRect = RectD(x: 0, y: 0, width: 1440, height: 900)
    private var calibrationTarget: Vec2?
    private var calibrating = false
    private var recordedSamples = 0

    private var gazeAssist = Vec2.zero
    private var lastGaze: GazeFeatures?
    private var lastFrameTime: Double = 0
    private var visionMilliseconds: Double = 0
    private var lastPublish: Double = 0
    private var lastHoverSample: Double = -10
    private var lastSpeed: Double = 0
    private var gazeActive = false
    private var smoothedLuma: Double = 0.5
    private var quality: Double = 1
    private var currentTargetFPS: Double = 60
    private var lastFrameRateChange: Double = -100

    private var gazeSmoothing: SmoothingConfig = {
        var config = SmoothingConfig()
        config.minCutoff = 0.9
        config.beta = 0.01
        config.predictionMs = 0
        config.deadband = 0
        config.precisionGain = 1
        config.outlierGate = 1e9
        return config
    }()

    var onSnapshot: ((TrackingSnapshot) -> Void)?
    var onCalibrationSample: ((CalibrationSample) -> Void)?

    var previewSession: AVCaptureSession { capture.session }

    init() {
        capture.onFrame = { [weak self] buffer, time in
            self?.process(buffer, time)
        }
    }

    // MARK: Control

    func start() -> String? {
        if let error = capture.start() {
            return error
        }
        cursor.syncPosition()
        pointer.reset(at: cursor.position)
        landmarkSmoother.reset()
        enhancer.reset()
        return nil
    }

    func stop() {
        capture.stop()
        gestures.reset()
        pointer.reset(at: nil)
        clickLock.reset()
        jitterMeter.reset()
        gazeFilter.reset()
        landmarkSmoother.reset()
        enhancer.reset()
        cursor.releaseButtons()
    }

    func update(settings: AppSettings) {
        stateLock.lock()
        self.settings = settings
        stateLock.unlock()
        // The sensor mode is fixed when the session is configured, so this takes effect on restart.
        capture.preferHighResolution = settings.preferHighResolution
    }

    func update(model: PointerModel?) {
        stateLock.lock()
        self.model = model
        stateLock.unlock()
        pointer.reset(at: nil)
    }

    func setScreen(_ rect: RectD) {
        stateLock.lock()
        let changed = rect != screenRect
        screenRect = rect
        stateLock.unlock()
        if changed { pointer.reset(at: nil) }
    }

    func currentScreenRect() -> RectD {
        stateLock.lock()
        defer { stateLock.unlock() }
        return screenRect
    }

    func setCalibration(target: Vec2?, active: Bool) {
        stateLock.lock()
        calibrationTarget = target
        calibrating = active
        if active { recordedSamples = 0 }
        stateLock.unlock()
        if active {
            pointer.reset(at: cursor.position)
            gestures.reset()
            clickLock.reset()
        }
    }

    func noteSampleRecorded() {
        stateLock.lock()
        recordedSamples += 1
        stateLock.unlock()
    }

    func recenter() {
        cursor.syncPosition()
        pointer.reset(at: cursor.position)
    }

    var cursorPosition: Vec2 { cursor.position }

    // MARK: Per-frame

    private func process(_ pixelBuffer: CVPixelBuffer, _ time: Double) {
        stateLock.lock()
        let config = settings
        let activeModel = model
        let target = calibrationTarget
        let isCalibrating = calibrating
        let screen = screenRect
        let samplesSoFar = recordedSamples
        stateLock.unlock()

        // 1. Light measurement and GPU conditioning.
        let luma = enhancer.measureLuma(pixelBuffer)
        smoothedLuma = smoothedLuma == 0 ? luma : smoothedLuma * 0.9 + luma * 0.1
        let decision = LowLightPolicy.decision(forLuma: smoothedLuma)
        let enhance = config.lowLightEnhancement && decision.enhance
        let analysedBuffer: CVPixelBuffer
        if enhance, let enhanced = enhancer.enhance(pixelBuffer, decision: decision) {
            analysedBuffer = enhanced
        } else {
            analysedBuffer = pixelBuffer
        }

        if config.adaptiveFrameRate,
           abs(decision.targetFPS - currentTargetFPS) >= 1,
           time - lastFrameRateChange > 5 {
            currentTargetFPS = decision.targetFPS
            lastFrameRateChange = time
            capture.setFrameRate(decision.targetFPS)
        }

        // 2. Vision (hand pose runs on the Neural Engine).
        let started = CACurrentMediaTime()
        let (rawHand, faceFrame) = vision.process(pixelBuffer: analysedBuffer, time: time)
        let elapsed = (CACurrentMediaTime() - started) * 1000
        visionMilliseconds = visionMilliseconds == 0 ? elapsed : visionMilliseconds * 0.85 + elapsed * 0.15

        if let gaze = faceFrame?.gaze {
            lastGaze = gaze
        }

        guard let rawHand else {
            if cursor.isLeftButtonDown { cursor.leftUp() }
            gestures.reset()
            clickLock.reset()
            landmarkSmoother.reset()
            gazeAssist = .zero
            publish(hand: nil, gesture: nil, output: nil, samples: samplesSoFar, enhancing: enhance)
            return
        }

        // 3. Per-landmark filtering, then re-derive every measurement from the smoothed joints.
        let smoothed = landmarkSmoother.smooth(rawHand.points,
                                               confidences: rawHand.confidences,
                                               time: time)
        guard let hand = HandFrame.build(points: smoothed, confidences: rawHand.confidences) else {
            publish(hand: nil, gesture: nil, output: nil, samples: samplesSoFar, enhancing: enhance)
            return
        }

        let lightQuality = PointerPipeline.smoothstep(smoothedLuma, 0.10, 0.38)
        quality = min(max(0.7 * hand.confidence + 0.3 * lightQuality, 0), 1)

        let dt = lastFrameTime == 0 ? 1.0 / 60.0 : max(min(time - lastFrameTime, 0.2), 1.0 / 1000.0)
        lastFrameTime = time

        let gesture = gestures.update(thumbContact: hand.thumbContact,
                                      pinchIndex: hand.pinchIndex,
                                      pinchMiddle: hand.pinchMiddle,
                                      fistRatio: hand.fistRatio,
                                      time: time,
                                      config: config.gestures)

        // One anchor for both hovering and pinching. Swapping to the thumb/index midpoint on click
        // used to slide the pointer while the user was still aiming; the anchor now sits on the
        // middle phalanx, which the pinch does not move, and click lock absorbs what is left.
        let anchor = hand.anchor

        // Before calibration there is no model, so fall back to a plain mirrored, aspect-corrected
        // mapping. It is rough, but it tracks the hand instead of pinning the pointer to a corner,
        // which is what the old fallback did once coordinates became aspect-corrected.
        let aspect = max(vision.lastAspect, 0.1)
        let rough = Vec2(min(max(1 - anchor.x / aspect, 0), 1),
                         min(max(1 - anchor.y, 0), 1))
        var normalized = activeModel?.normalized(for: anchor, pose: hand.pose) ?? rough

        // Gaze assist: a coarse absolute prior that removes drift. Three things keep it helpful
        // rather than destructive — its gain follows the model's own cross-validated accuracy, the
        // clamp follows the same figure (a shaky model can only nudge), and it is gated on the hand
        // settling, because that is when precision matters and when a slow pull cannot smear
        // intentional motion.
        if config.eyeAssistEnabled, let gazeModel = activeModel?.gaze, let gaze = lastGaze,
           gaze.openness > 0.30, let rawGaze = gazeModel.predict(gaze) {
            let smoothedGaze = gazeFilter.filter(rawGaze, time: time, config: gazeSmoothing)
            let accuracy = min(max(gazeModel.rmsPixels, 6), 200)
            let accuracyWeight = exp(-accuracy / 90)          // 1.0 at 6 px, 0.33 at 100 px
            let settle = 1 - PointerPipeline.smoothstep(lastSpeed, 60, 240)
            let gain = config.eyeAssistStrength * accuracyWeight * settle
            let limit = min(0.055, 2.4 * accuracy / max(screen.width, 1))
            let offset = smoothedGaze - normalized
            let desired = Vec2(min(max(offset.x * gain, -limit), limit),
                               min(max(offset.y * gain, -limit), limit))
            gazeAssist = Vec2.lerp(gazeAssist, desired, 1 - exp(-dt / 0.55))
            gazeActive = true
        } else {
            gazeAssist = Vec2.lerp(gazeAssist, .zero, 1 - exp(-dt / 0.35))
            gazeActive = false
        }
        normalized = normalized + gazeAssist

        let mapped = screen.denormalized(normalized)

        // Click lock: absorb the few centimetres a pinch drags the fingertip.
        if gesture.events.contains(where: { if case .leftDown = $0 { return true }; return false }) {
            clickLock.begin(offset: mapped - cursor.position)
        }
        clickLock.decay(dt: dt)
        let compensated = clickLock.apply(mapped)

        // Poor tracking quality widens the deadband instead of letting noise through.
        var pipelineConfig = config.smoothing
        pipelineConfig.deadband *= (1 + (1 - quality) * 0.9)

        guard let output = pointer.process(modelPoint: compensated,
                                           time: time,
                                           bounds: screen,
                                           config: pipelineConfig) else {
            publish(hand: hand, gesture: gesture, output: nil, samples: samplesSoFar, enhancing: enhance)
            return
        }

        cursor.move(to: output.point, dragging: gesture.dragging || cursor.isLeftButtonDown)
        for event in gesture.events {
            switch event {
            case .leftDown(let clickState): cursor.leftDown(clickState: clickState)
            case .leftUp: cursor.leftUp()
            case .rightDown: cursor.rightDown()
            case .rightUp: cursor.rightUp()
            }
        }

        jitterMeter.record(output.point, time: time)
        lastSpeed = output.velocity.length

        // Calibration capture: a pinch is a precise sample; a settled hover is a cheap extra one.
        // Both are weighted by tracking quality, so a noisy dark frame cannot skew the fit.
        if isCalibrating, let targetPoint = target {
            let qualityWeight = 0.5 + 0.5 * quality
            let pinchDown = gesture.events.contains { if case .leftDown = $0 { return true }; return false }
            if pinchDown {
                deliver(CalibrationSample(source: anchor, target: targetPoint, weight: 2.0 * qualityWeight,
                                          group: 0, pose: hand.pose, gaze: lastGaze))
                lastHoverSample = time
            } else if config.hoverSamplesEnabled {
                let targetPixels = screen.denormalized(targetPoint)
                let stationary = output.velocity.length < 40
                if output.point.distance(to: targetPixels) < 80, stationary, time - lastHoverSample > 0.22 {
                    lastHoverSample = time
                    deliver(CalibrationSample(source: anchor, target: targetPoint, weight: 0.55 * qualityWeight,
                                              group: 0, pose: hand.pose, gaze: lastGaze))
                }
            }
        }

        publish(hand: hand, gesture: gesture, output: output, samples: samplesSoFar, enhancing: enhance)
    }

    private func deliver(_ sample: CalibrationSample) {
        onCalibrationSample?(sample)
    }

    private func publish(hand: HandFrame?,
                         gesture: GestureOutput?,
                         output: PointerPipelineOutput?,
                         samples: Int,
                         enhancing: Bool) {
        let now = CACurrentMediaTime()
        guard now - lastPublish > 0.04 else { return }
        lastPublish = now

        var snapshot = TrackingSnapshot()
        snapshot.isRunning = true
        snapshot.fps = capture.framesPerSecond
        snapshot.targetFPS = currentTargetFPS
        snapshot.visionMilliseconds = visionMilliseconds
        snapshot.jitter = jitterMeter.jitterRMS
        snapshot.cursor = output?.point ?? cursor.position
        snapshot.quality = quality
        snapshot.luma = smoothedLuma
        snapshot.enhancing = enhancing
        snapshot.cameraWidth = capture.activeWidth
        snapshot.cameraHeight = capture.activeHeight
        snapshot.gazeActive = gazeActive
        snapshot.gazeOpenness = lastGaze?.openness ?? 0
        snapshot.sampleCount = samples

        if let hand {
            snapshot.hasHand = true
            snapshot.handPoints = hand.points
            snapshot.anchor = hand.anchor
            snapshot.pinchIndex = hand.pinchIndex
            snapshot.pinchMiddle = hand.pinchMiddle
            snapshot.thumbContact = hand.thumbContact
            snapshot.fist = hand.fistRatio
        }
        if let gesture {
            snapshot.dragging = gesture.dragging
            snapshot.frozen = gesture.frozen
        }
        snapshot.hasFace = lastGaze != nil
        snapshot.gaze = gazeAssist.length > 0.0005 ? gazeAssist : nil

        stateLock.lock()
        snapshot.calibrated = model != nil
        snapshot.gazeAccuracy = model?.gaze?.rmsPixels ?? 0
        stateLock.unlock()

        DispatchQueue.main.async { [weak self] in
            self?.onSnapshot?(snapshot)
        }
    }
}
