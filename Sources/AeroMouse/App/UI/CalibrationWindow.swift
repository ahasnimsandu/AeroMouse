import Foundation
import AppKit
import SwiftUI

/// Full-screen borderless overlay used for calibration targets. It is deliberately a real window
/// so the target can sit anywhere on the display while the system cursor does the pointing.
@MainActor
final class CalibrationWindowController {
    private var window: NSWindow?

    func show(with app: AppState, screen: NSScreen?) {
        close()
        let frame = screen?.frame ?? NSScreen.main?.frame ?? CGRect(x: 0, y: 0, width: 1280, height: 800)
        let window = OverlayWindow(contentRect: frame,
                                   styleMask: [.borderless],
                                   backing: .buffered,
                                   defer: false)
        window.level = .screenSaver
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = false
        window.ignoresMouseEvents = false
        window.contentView = NSHostingView(rootView: CalibrationOverlayView().environmentObject(app))
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        self.window = window
    }

    func close() {
        window?.orderOut(nil)
        window?.close()
        window = nil
    }
}

final class OverlayWindow: NSWindow {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
}
