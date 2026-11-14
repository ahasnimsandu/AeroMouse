import Foundation
import Vision
import CoreVideo
import CoreGraphics

/// One frame of hand tracking, reduced to the descriptors the pointer model and gestures need.
struct HandFrame {
    var points: [VNHumanHandPoseObservation.JointName: Vec2] = [:]
    var confidences: [VNHumanHandPoseObservation.JointName: Double] = [:]

    /// The single pointer anchor: middle phalanx of the index finger, in aspect-corrected image
    /// coordinates. It is deliberately *not* the fingertip — the tip travels a centimetre or two
    /// when the finger curls into a pinch, and the anchor must not move when that happens.
    var anchor: Vec2 = .zero

    var pose = PointerPose()
    var pinchIndex: Double = .infinity
    var pinchMiddle: Double = .infinity
    /// Thumb pad to the *edge* of the index finger, in hand units. This is the Apple Watch tap
    /// geometry: the two fingers meet along their sides, anywhere from the first knuckle to the tip.
    var thumbContact: Double = .infinity
    var fistRatio: Double = 0
    var confidence: Double = 0

    /// Derives every measurement from a set of joint positions. Called on raw landmarks and again
    /// after smoothing, so both paths agree exactly.
    static func build(points: [VNHumanHandPoseObservation.JointName: Vec2],
                      confidences: [VNHumanHandPoseObservation.JointName: Double]) -> HandFrame? {
        func point(_ name: VNHumanHandPoseObservation.JointName) -> Vec2? { points[name] }
        func conf(_ name: VNHumanHandPoseObservation.JointName) -> Double { confidences[name] ?? 0 }

        // Only joints the pointer genuinely needs are required. Anything else (the outer fingers)
        // is optional, because a hand at the edge of frame or in poor light routinely loses them,
        // and a dropped frame used to kill tracking entirely.
        guard let wrist = point(.wrist),
              let middleMCP = point(.middleMCP),
              let indexMCP = point(.indexMCP),
              let indexPIP = point(.indexPIP),
              let indexDIP = point(.indexDIP),
              let indexTip = point(.indexTip),
              let thumbTip = point(.thumbTip) else {
            return nil
        }

        let handScale = wrist.distance(to: middleMCP)
        guard handScale > 0.008, handScale < 1.5 else { return nil }

        var frame = HandFrame(points: points, confidences: confidences)

        // Wrist roll tells the model how rotated the palm is; a homography cannot know this.
        // Palm width is used for spread rather than the gap between fingertips: it is a rigid
        // distance across the knuckles, so it does not move when the fingers curl.
        let spread: Double
        if let littleMCP = point(.littleMCP) {
            spread = indexMCP.distance(to: littleMCP) / handScale
        } else if let littleTip = point(.littleTip) {
            spread = littleTip.distance(to: indexTip) / handScale
        } else {
            spread = 1
        }

        frame.pose = PointerPose(handScale: handScale,
                                roll: atan2(middleMCP.y - wrist.y, middleMCP.x - wrist.x),
                                spread: spread)

        // Measured to the segment, not to the fingertip, so a tap on the side of the finger registers
        // as easily as one at the tip — and so the reading does not change when the finger curls.
        frame.thumbContact = Vec2.distance(thumbTip, toSegment: indexPIP, indexTip) / handScale
        frame.pinchIndex = thumbTip.distance(to: indexTip) / handScale
        frame.pinchMiddle = point(.middleTip).map { thumbTip.distance(to: $0) / handScale } ?? .infinity

        // Fist detection: how far each visible fingertip has curled back to its knuckle.
        let fingers: [(VNHumanHandPoseObservation.JointName, VNHumanHandPoseObservation.JointName)] = [
            (.indexTip, .indexMCP), (.middleTip, .middleMCP),
            (.ringTip, .ringMCP), (.littleTip, .littleMCP),
        ]
        var curl = 0.0
        var counted = 0
        for (tipName, knuckleName) in fingers {
            guard let tip = point(tipName), let knuckle = point(knuckleName) else { continue }
            let ratio = tip.distance(to: knuckle) / handScale
            curl += 1 - min(max((ratio - 0.18) / 0.75, 0), 1)
            counted += 1
        }
        frame.fistRatio = counted > 0 ? curl / Double(counted) : 0

        // The pointer anchor: halfway between the first and second index knuckle. Hovering,
        // clicking and calibration all read this one point, so aiming and clicking cannot disagree.
        frame.anchor = Vec2.lerp(indexDIP, indexPIP, 0.5)

        // Mean confidence over the joints that matter. Taking the minimum let one unlucky knuckle
        // report a quality score near zero on a perfectly good frame.
        let keyJoints: [VNHumanHandPoseObservation.JointName] = [
            .wrist, .middleMCP, .indexMCP, .indexPIP, .indexDIP, .indexTip, .thumbTip,
        ]
        frame.confidence = keyJoints.map { conf($0) }.reduce(0, +) / Double(keyJoints.count)
        return frame
    }
}

struct FaceFrame {
    var gaze: GazeFeatures
    var confidence: Double
}

/// Runs Vision hand-pose every frame and face landmarks every third frame (gaze is a slow signal,
/// and the hand request is the latency-critical one). Vision executes the hand-pose network on the
/// Neural Engine, so this is already the NPU path.
final class VisionPipeline {
    private let handRequest = VNDetectHumanHandPoseRequest()
    private let faceRequest = VNDetectFaceLandmarksRequest()
    private var frameIndex = 0
    private(set) var lastError: String?
    private(set) var lastAspect: Double = 4.0 / 3.0

    func process(pixelBuffer: CVPixelBuffer,
                 time: Double) -> (hand: HandFrame?, face: FaceFrame?) {
        frameIndex += 1
        var requests: [VNRequest] = [handRequest]
        let runFace = (frameIndex % 3) == 0
        if runFace { requests.append(faceRequest) }

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        do {
            try handler.perform(requests)
        } catch {
            lastError = error.localizedDescription
            return (nil, nil)
        }

        let width = Double(CVPixelBufferGetWidth(pixelBuffer))
        let height = Double(CVPixelBufferGetHeight(pixelBuffer))
        lastAspect = height > 0 ? max(width / height, 0.1) : 4.0 / 3.0

        let hand = makeHandFrame(from: handRequest.results, aspect: lastAspect)
        let face = runFace ? makeFaceFrame(from: faceRequest.results) : nil
        return (hand, face)
    }

    // MARK: Hand

    private func makeHandFrame(from observations: [VNHumanHandPoseObservation]?,
                               aspect: Double) -> HandFrame? {
        guard let observation = observations?.first else { return nil }
        guard let recognized = try? observation.recognizedPoints(.all) else { return nil }

        var points: [VNHumanHandPoseObservation.JointName: Vec2] = [:]
        var confidences: [VNHumanHandPoseObservation.JointName: Double] = [:]
        for (name, point) in recognized where point.confidence > 0.1 {
            // Vision normalizes x and y independently, so a 4:3 frame squashes the hand
            // horizontally. Scaling x by the aspect ratio restores a square-pixel space: distances,
            // pinch ratios and wrist roll are then geometrically correct instead of depending on
            // which way the hand happens to point.
            points[name] = Vec2(Double(point.location.x) * aspect, Double(point.location.y))
            confidences[name] = Double(point.confidence)
        }

        return HandFrame.build(points: points, confidences: confidences)
    }

    // MARK: Face

    private func makeFaceFrame(from observations: [VNFaceObservation]?) -> FaceFrame? {
        guard let face = observations?.first, let landmarks = face.landmarks else { return nil }
        let unit = CGSize(width: 1, height: 1)
        let aspect = max(lastAspect, 0.1)

        guard let leftEye = landmarks.leftEye?.pointsInImage(imageSize: unit),
              let rightEye = landmarks.rightEye?.pointsInImage(imageSize: unit),
              !leftEye.isEmpty, !rightEye.isEmpty else { return nil }

        func bounds(_ points: [CGPoint]) -> CGRect {
            let xs = points.map { $0.x }
            let ys = points.map { $0.y }
            let minX = xs.min() ?? 0
            let maxX = xs.max() ?? 0
            let minY = ys.min() ?? 0
            let maxY = ys.max() ?? 0
            return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
        }

        let leftBox = bounds(leftEye)
        let rightBox = bounds(rightEye)
        guard leftBox.width > 0.0015, leftBox.height > 0.0015,
              rightBox.width > 0.0015, rightBox.height > 0.0015 else { return nil }

        // Iris position inside each eye opening: the core gaze signal on a camera without depth.
        func irisOffset(_ pupil: CGPoint?, _ box: CGRect) -> (x: Double, y: Double)? {
            guard let pupil, box.width > 1e-6, box.height > 1e-6 else { return nil }
            return ((Double(pupil.x) - box.midX) / box.width,
                    (Double(pupil.y) - box.midY) / box.height)
        }

        let left = irisOffset(landmarks.leftPupil?.pointsInImage(imageSize: unit).first, leftBox)
        let right = irisOffset(landmarks.rightPupil?.pointsInImage(imageSize: unit).first, rightBox)
        // One visible pupil still carries usable gaze; only a fully missing gaze is rejected.
        guard let primary = left ?? right else { return nil }
        let secondary = left == nil ? primary : (right ?? primary)

        // Eye opening relative to its own width: collapses during a blink, which is where most
        // bad gaze samples come from. Width is aspect-corrected so the ratio is physical.
        let eyeWidth = max(leftBox.width * aspect, 1e-6)
        let aspectRatio = leftBox.height / eyeWidth
        let openness = min(max((aspectRatio - 0.12) / 0.18, 0), 1)

        let box = face.boundingBox
        let gaze = GazeFeatures(irisX: (primary.x + secondary.x) / 2,
                                irisY: (primary.y + secondary.y) / 2,
                                yaw: face.yaw?.doubleValue ?? 0,
                                roll: face.roll?.doubleValue ?? 0,
                                leftX: left?.x ?? secondary.x,
                                leftY: left?.y ?? secondary.y,
                                rightX: right?.x ?? primary.x,
                                rightY: right?.y ?? primary.y,
                                pitch: face.pitch?.doubleValue ?? 0,
                                headX: Double(box.midX),
                                headY: Double(box.midY),
                                headScale: Double(box.width) * aspect,
                                openness: openness)
        return FaceFrame(gaze: gaze, confidence: Double(face.confidence))
    }
}
