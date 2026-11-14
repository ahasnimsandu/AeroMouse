import Foundation
import Vision
import CoreVideo
import CoreGraphics

/// One frame of hand tracking, reduced to the descriptors the pointer model and gestures need.
struct HandFrame {
    var points: [VNHumanHandPoseObservation.JointName: Vec2] = [:]
    var confidences: [VNHumanHandPoseObservation.JointName: Double] = [:]
    var anchorOpen: Vec2 = .zero     // stable open-hand pointer anchor
    var anchorPinch: Vec2 = .zero    // thumb/index midpoint while pinching
    var pose = PointerPose()
    var pinchIndex: Double = .infinity
    var pinchMiddle: Double = .infinity
    var fistRatio: Double = 0
    var confidence: Double = 0

    /// Derives every measurement from a set of joint positions. Called on raw landmarks and again
    /// after smoothing, so both paths agree exactly.
    static func build(points: [VNHumanHandPoseObservation.JointName: Vec2],
                      confidences: [VNHumanHandPoseObservation.JointName: Double]) -> HandFrame? {
        func point(_ name: VNHumanHandPoseObservation.JointName) -> Vec2? { points[name] }
        func conf(_ name: VNHumanHandPoseObservation.JointName) -> Double { confidences[name] ?? 0 }

        guard let wrist = point(.wrist),
              let middleMCP = point(.middleMCP),
              let indexMCP = point(.indexMCP),
              let indexDIP = point(.indexDIP),
              let indexTip = point(.indexTip),
              let thumbTip = point(.thumbTip),
              let middleTip = point(.middleTip),
              let ringTip = point(.ringTip),
              let littleTip = point(.littleTip),
              let ringMCP = point(.ringMCP),
              let littleMCP = point(.littleMCP) else {
            return nil
        }

        let handScale = wrist.distance(to: middleMCP)
        guard handScale > 0.008, handScale < 1.5 else { return nil }

        var frame = HandFrame(points: points, confidences: confidences)

        // Wrist roll tells the model how rotated the palm is; a homography cannot know this.
        frame.pose = PointerPose(
            handScale: handScale,
            roll: atan2(middleMCP.y - wrist.y, middleMCP.x - wrist.x),
            spread: littleTip.distance(to: indexTip) / handScale
        )

        frame.pinchIndex = thumbTip.distance(to: indexTip) / handScale
        frame.pinchMiddle = thumbTip.distance(to: middleTip) / handScale

        // Fist detection: how close each fingertip has curled back to its knuckle.
        let fingers: [(Vec2, Vec2)] = [(indexTip, indexMCP), (middleTip, middleMCP),
                                       (ringTip, ringMCP), (littleTip, littleMCP)]
        var curl = 0.0
        for (tip, knuckle) in fingers {
            let ratio = tip.distance(to: knuckle) / handScale
            curl += 1 - min(max((ratio - 0.18) / 0.75, 0), 1)
        }
        frame.fistRatio = curl / Double(fingers.count)

        // Open anchor: weighted back from the very tip, which is the jitteriest landmark.
        frame.anchorOpen = Vec2.lerp(indexTip, indexDIP, 0.45)
        frame.anchorPinch = (thumbTip + indexTip) * 0.5

        frame.confidence = min(conf(.indexTip), min(conf(.indexDIP),
                              min(conf(.thumbTip), min(conf(.middleMCP), conf(.wrist)))))
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

        let hand = makeHandFrame(from: handRequest.results)
        let face = runFace ? makeFaceFrame(from: faceRequest.results) : nil
        return (hand, face)
    }

    // MARK: Hand

    private func makeHandFrame(from observations: [VNHumanHandPoseObservation]?) -> HandFrame? {
        guard let observation = observations?.first else { return nil }
        guard let recognized = try? observation.recognizedPoints(.all) else { return nil }

        var points: [VNHumanHandPoseObservation.JointName: Vec2] = [:]
        var confidences: [VNHumanHandPoseObservation.JointName: Double] = [:]
        for (name, point) in recognized where point.confidence > 0.1 {
            points[name] = Vec2(Double(point.location.x), Double(point.location.y))
            confidences[name] = Double(point.confidence)
        }

        return HandFrame.build(points: points, confidences: confidences)
    }

    // MARK: Face

    private func makeFaceFrame(from observations: [VNFaceObservation]?) -> FaceFrame? {
        guard let face = observations?.first, let landmarks = face.landmarks else { return nil }
        let unit = CGSize(width: 1, height: 1)

        guard let leftPupil = landmarks.leftPupil?.pointsInImage(imageSize: unit).first,
              let rightPupil = landmarks.rightPupil?.pointsInImage(imageSize: unit).first,
              let leftEye = landmarks.leftEye?.pointsInImage(imageSize: unit),
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

        // Iris position inside the eye opening: the core gaze signal on a camera without depth.
        let leftOffsetX = (Double(leftPupil.x) - leftBox.midX) / leftBox.width
        let leftOffsetY = (Double(leftPupil.y) - leftBox.midY) / leftBox.height
        let rightOffsetX = (Double(rightPupil.x) - rightBox.midX) / rightBox.width
        let rightOffsetY = (Double(rightPupil.y) - rightBox.midY) / rightBox.height

        let gaze = GazeFeatures(irisX: (leftOffsetX + rightOffsetX) / 2,
                                irisY: (leftOffsetY + rightOffsetY) / 2,
                                yaw: face.yaw?.doubleValue ?? 0,
                                roll: face.roll?.doubleValue ?? 0)
        return FaceFrame(gaze: gaze, confidence: Double(face.confidence))
    }
}
