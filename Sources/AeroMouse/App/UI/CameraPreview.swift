import SwiftUI
import AppKit
import AVFoundation
import Vision
import QuartzCore

// MARK: - Camera preview

struct CameraPreview: NSViewRepresentable {
    let session: AVCaptureSession
    let mirrored: Bool

    func makeNSView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        view.applyMirroring(mirrored)
        return view
    }

    func updateNSView(_ nsView: PreviewView, context: Context) {
        if nsView.previewLayer.session !== session {
            nsView.previewLayer.session = session
        }
        nsView.applyMirroring(mirrored)
    }

    final class PreviewView: NSView {
        let previewLayer = AVCaptureVideoPreviewLayer()

        override init(frame frameRect: NSRect) {
            super.init(frame: frameRect)
            wantsLayer = true
            layer?.backgroundColor = NSColor.black.cgColor
            previewLayer.frame = bounds
            layer?.addSublayer(previewLayer)
        }

        required init?(coder: NSCoder) { fatalError("not used") }

        override func layout() {
            super.layout()
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            previewLayer.frame = bounds
            CATransaction.commit()
        }

        func applyMirroring(_ mirrored: Bool) {
            guard let connection = previewLayer.connection else { return }
            connection.automaticallyAdjustsVideoMirroring = false
            connection.isVideoMirrored = mirrored
        }
    }
}

// MARK: - Landmark overlay

struct HandOverlay: View {
    let snapshot: TrackingSnapshot
    let mirrored: Bool

    private static let bones: [(VNHumanHandPoseObservation.JointName, VNHumanHandPoseObservation.JointName)] = [
        (.wrist, .thumbCMC), (.thumbCMC, .thumbMP), (.thumbMP, .thumbIP), (.thumbIP, .thumbTip),
        (.wrist, .indexMCP), (.indexMCP, .indexPIP), (.indexPIP, .indexDIP), (.indexDIP, .indexTip),
        (.wrist, .middleMCP), (.middleMCP, .middlePIP), (.middlePIP, .middleDIP), (.middleDIP, .middleTip),
        (.wrist, .ringMCP), (.ringMCP, .ringPIP), (.ringPIP, .ringDIP), (.ringDIP, .ringTip),
        (.wrist, .littleMCP), (.littleMCP, .littlePIP), (.littlePIP, .littleDIP), (.littleDIP, .littleTip),
        (.indexMCP, .middleMCP), (.middleMCP, .ringMCP), (.ringMCP, .littleMCP),
    ]

    var body: some View {
        Canvas { context, size in
            guard snapshot.hasHand else { return }

            func project(_ point: Vec2) -> CGPoint {
                let x = mirrored ? 1 - point.x : point.x
                return CGPoint(x: x * size.width, y: (1 - point.y) * size.height)
            }

            for (a, b) in Self.bones {
                guard let pa = snapshot.handPoints[a], let pb = snapshot.handPoints[b] else { continue }
                var path = Path()
                path.move(to: project(pa))
                path.addLine(to: project(pb))
                context.stroke(path, with: .color(.green.opacity(0.75)), lineWidth: 2)
            }

            for (_, point) in snapshot.handPoints {
                let center = project(point)
                let rect = CGRect(x: center.x - 2.5, y: center.y - 2.5, width: 5, height: 5)
                context.fill(Path(ellipseIn: rect), with: .color(.white.opacity(0.9)))
            }

            if let anchor = snapshot.anchor {
                let center = project(anchor)
                let rect = CGRect(x: center.x - 5, y: center.y - 5, width: 10, height: 10)
                context.stroke(Path(ellipseIn: rect), with: .color(.red), lineWidth: 2)
            }
        }
        .allowsHitTesting(false)
    }
}
