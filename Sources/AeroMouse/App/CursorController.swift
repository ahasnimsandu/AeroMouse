import Foundation
import CoreGraphics
import ApplicationServices
import AppKit

/// Posts real mouse events. Moving the pointer programmatically requires Accessibility
/// permission; without it macOS silently drops the events.
final class CursorController {
    private let source = CGEventSource(stateID: .hidSystemState)
    private(set) var position: Vec2 = .zero
    private(set) var isLeftButtonDown = false
    private var clickState = 1

    var isTrusted: Bool { AXIsProcessTrusted() }

    init() {
        syncPosition()
    }

    func syncPosition() {
        if let event = CGEvent(source: nil) {
            position = Vec2(event.location)
        }
    }

    func move(to point: Vec2, dragging: Bool) {
        let target = CGPoint(x: point.x.rounded(), y: point.y.rounded())
        if abs(target.x - position.x) < 0.5, abs(target.y - position.y) < 0.5 { return }
        position = Vec2(target)
        let type: CGEventType = dragging ? .leftMouseDragged : .mouseMoved
        guard let event = CGEvent(mouseEventSource: source,
                                  mouseType: type,
                                  mouseCursorPosition: target,
                                  mouseButton: .left) else { return }
        if dragging {
            event.setIntegerValueField(.mouseEventButtonNumber, value: 0)
        }
        event.post(tap: .cghidEventTap)
    }

    func leftDown(clickState: Int) {
        self.clickState = max(clickState, 1)
        isLeftButtonDown = true
        post(.leftMouseDown, button: .left, clickState: self.clickState)
    }

    func leftUp() {
        guard isLeftButtonDown else { return }
        isLeftButtonDown = false
        post(.leftMouseUp, button: .left, clickState: clickState)
        clickState = 1
    }

    func rightDown() { post(.rightMouseDown, button: .right, clickState: 1) }
    func rightUp() { post(.rightMouseUp, button: .right, clickState: 1) }

    func releaseButtons() {
        if isLeftButtonDown { leftUp() }
    }

    private func post(_ type: CGEventType, button: CGMouseButton, clickState: Int) {
        guard let event = CGEvent(mouseEventSource: source,
                                  mouseType: type,
                                  mouseCursorPosition: position.cg,
                                  mouseButton: button) else { return }
        event.setIntegerValueField(.mouseEventClickState, value: Int64(clickState))
        event.post(tap: .cghidEventTap)
    }
}
