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
    var tapOnLevel: Double = 0
    var tapOffLevel: Double = 0
    var contactBaseline: Double = 0
    var tapClosed = false
    var scrolling = false
    var flowLocked = false
    var tapTrainingActive = false
    var tapTrainingProgress: Double = 0
    var tapSuggestion: TapTrainer.Suggestion?
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
    private let fingerFlow = FingerFlow()

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
    private var scrollAccumulator = ScrollAccumulator()
    private var lastScrollY: Double?
    private var handLostTime: Double = 0
    private var lastOutputVelocity: Vec2 = .zero

    // Tap trainer. The camera queue owns the samples; the flag and result are exchanged under the lock.
    private var tapTrainingActive = false
    private var tapTrainingStartedAt: Double = 0
    private var tapTrainingContacts: [Double] = []
    private var tapSuggestion: TapTrainer.Suggestion?
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
        fingerFlow.reset()
        return nil
    }

    func stop() {
        capture.stop()
        gestures.reset()
        pointer.reset(at: nil)
        clickLock.reset()
        jitterMeter.reset()
        scrollAccumulator.reset()
        lastScrollY = nil
        gazeFilter.reset()
        landmarkSmoother.reset()
        enhancer.reset()
        fingerFlow.reset()
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

    // MARK: Tap training

    func beginTapTraining() {
        stateLock.lock()
        tapTrainingActive = true
        tapSuggestion = nil
        tapTrainingStartedAt = 0
        stateLock.unlock()
        gestures.reset()
    }

    func cancelTapTraining() {
        stateLock.lock()
        tapTrainingActive = false
        tapTrainingStartedAt = 0
        tapSuggestion = nil
        stateLock.unlock()
    }

    /// Called once the UI has applied a suggestion, so it is not applied twice.
    func clearTapSuggestion() {
        stateLock.lock()
        tapSuggestion = nil
        stateLock.unlock()
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
        let training = tapTrainingActive
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
            cursor.releaseButtons()
            gestures.reset()
            clickLock.reset()
            landmarkSmoother.reset()
            fingerFlow.reset()
            gazeAssist = .zero
            if handLostTime == 0 { handLostTime = time }
            let elapsed = time - handLostTime
            if elapsed < 0.35, lastOutputVelocity.length > 5 {
                let dt = lastFrameTime == 0 ? 1.0 / 60.0 : max(min(time - lastFrameTime, 0.2), 1.0 / 1000.0)
                lastFrameTime = time
                let decay = exp(-elapsed / 0.18)
                let v = lastOutputVelocity * decay
                let candidate = cursor.position + v * dt
                cursor.move(to: candidate.clamped(to: screen), dragging: false)
                publish(hand: nil, gesture: nil, output: nil, samples: samplesSoFar, enhancing: enhance)
                return
            }
            handLostTime = 0
            lastOutputVelocity = .zero
            publish(hand: nil, gesture: nil, output: nil, samples: samplesSoFar, enhancing: enhance)
            return
        }
        handLostTime = 0

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

        // Tap trainer: watch the contact reading for a few seconds while the user relaxes and taps.
        if training {
            if tapTrainingStartedAt == 0 { tapTrainingStartedAt = time }
            if hand.thumbContact.isFinite, hand.thumbContact > 0 {
                tapTrainingContacts.append(hand.thumbContact)
            }
            if time - tapTrainingStartedAt > 6.5 {
                let suggestion = TapTrainer.suggestion(from: tapTrainingContacts)
                stateLock.lock()
                tapTrainingActive = false
                tapSuggestion = suggestion
                stateLock.unlock()
                tapTrainingContacts.removeAll(keepingCapacity: true)
                tapTrainingStartedAt = 0
            }
        }

        let dt = lastFrameTime == 0 ? 1.0 / 60.0 : max(min(time - lastFrameTime, 0.2), 1.0 / 1000.0)
        lastFrameTime = time

        // While training, taps are data rather than input: no clicks may fire.
        var gestureConfig = config.gestures
        if training {
            gestureConfig.tapEnabled = false
            gestureConfig.pinchClickEnabled = false
        }
        let gesture = gestures.update(thumbContact: hand.thumbContact,
                                      pinchIndex: hand.pinchIndex,
                                      pinchMiddle: hand.pinchMiddle,
                                      indexMiddleGap: hand.indexMiddleGap,
                                      fistRatio: hand.fistRatio,
                                      time: time,
                                      config: gestureConfig)

        // One anchor for both hovering and pinching. Swapping to the thumb/index midpoint on click
        // used to slide the pointer while the user was still aiming; the anchor now sits on the
        // middle phalanx, which the pinch does not move, and click lock absorbs what is left.
        //
        // The flow tracker refines it: motion between frames is measurable to a fraction of a pixel,
        // far below the landmark's own noise, and the landmark keeps the patch from drifting.
        var anchor = hand.anchor
        if config.fingerFlowEnabled,
           let refined = fingerFlow.refine(anchor: anchor, buffer: analysedBuffer, time: time) {
            anchor = refined
        }


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
            let accuracyWeight = exp(-accuracy / 110)         // 1.0 at 6 px, 0.40 at 100 px
            let settle = 1 - PointerPipeline.smoothstep(lastSpeed, 90, 320)
            let gain = min(1.0, config.eyeAssistStrength * accuracyWeight * settle)
            let limit = min(0.09, 3.6 * accuracy / max(screen.width, 1))
            let offset = smoothedGaze - normalized
            let desired = Vec2(min(max(offset.x * gain, -limit), limit),
                               min(max(offset.y * gain, -limit), limit))
            gazeAssist = Vec2.lerp(gazeAssist, desired, 1 - exp(-dt / 0.45))
            gazeActive = true
        } else {
            gazeAssist = Vec2.lerp(gazeAssist, .zero, 1 - exp(-dt / 0.35))
            gazeActive = false
        }
        normalized = normalized + gazeAssist

        let mapped = screen.denormalized(normalized)

        // Scroll gesture: two fingers held together and moved vertically. Wheel events replace
        // pointer motion, so the cursor stays exactly where it was while you scroll.
        if gesture.scrolling {
            let y = mapped.y
            if let previous = lastScrollY, abs(y - previous) > config.gestures.scrollDeadzone {
                let ticks = scrollAccumulator.ticks(forPixels: (y - previous) * config.gestures.scrollSpeed)
                if ticks != 0 { cursor.scroll(lines: ticks) }
            }
            lastScrollY = y
            publish(hand: hand, gesture: gesture, output: nil, samples: samplesSoFar, enhancing: enhance)
            return
        }
        lastScrollY = nil
        scrollAccumulator.reset()

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
        lastOutputVelocity = output.velocity

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
        snapshot.flowLocked = fingerFlow.isLocked
        stateLock.lock()
        snapshot.tapTrainingActive = tapTrainingActive
        snapshot.tapSuggestion = tapSuggestion
        if tapTrainingActive, tapTrainingStartedAt > 0 {
            snapshot.tapTrainingProgress = min(1, max(0, (CACurrentMediaTime() - tapTrainingStartedAt) / 6.5))
        }
        stateLock.unlock()
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
            snapshot.tapOnLevel = gesture.tapOnLevel
            snapshot.tapOffLevel = gesture.tapOffLevel
            snapshot.contactBaseline = gesture.contactBaseline
            snapshot.tapClosed = gesture.tapClosed
            snapshot.scrolling = gesture.scrolling
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
