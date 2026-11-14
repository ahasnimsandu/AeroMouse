import Foundation

// MARK: - Gesture configuration

struct GestureConfig: Codable {
    /// Tap: the thumb pad meets the edge of the index finger, like Apple Watch double tap.
    /// Measured in hand units, so it does not depend on how big the hand or the frame is.
    var tapEnabled: Bool = true
    var tapOn: Double = 0.085         // contact is closed below this
    var tapOff: Double = 0.135        // ...and open again above it (hysteresis)
    var pinchClickEnabled: Bool = false
    var pinchOn: Double = 0.42        // thumb-index distance / hand scale to close
    var pinchOff: Double = 0.62       // ...and to open again (hysteresis gap kills flicker)
    var middlePinchOn: Double = 0.46
    var middlePinchOff: Double = 0.66
    var minDownFrames: Int = 2        // debounce: frames below threshold before a click registers
    var minUpFrames: Int = 2
    var doubleClickWindow: Double = 0.34
    var freezeOn: Double = 0.80       // closed fist
    var freezeOff: Double = 0.62
    var rightClickEnabled: Bool = true
    var doubleClickEnabled: Bool = true
    var freezeEnabled: Bool = true
    var dragEnabled: Bool = true
}

// MARK: - Events

enum PointerEvent: Equatable {
    case leftDown(clickState: Int)
    case leftUp
    case rightDown
    case rightUp
}

struct GestureOutput {
    var events: [PointerEvent] = []
    var dragging: Bool = false
    var pinching: Bool = false
    var frozen: Bool = false
}

// MARK: - Recognizer

/// Hysteresis + frame debouncing on every threshold. A hand hovering exactly on the pinch
/// boundary therefore produces one clean click rather than a burst of event flicker.
final class GestureRecognizer {
    private var leftDown = false
    private var leftDownFrames = 0
    private var leftUpFrames = 0
    private var rightDown = false
    private var rightDownFrames = 0
    private var rightUpFrames = 0
    private var frozen = false
    private var lastLeftUpTime = -100.0
    private var clickCount = 1

    func reset() {
        leftDown = false
        leftDownFrames = 0
        leftUpFrames = 0
        rightDown = false
        rightDownFrames = 0
        rightUpFrames = 0
        frozen = false
        lastLeftUpTime = -100.0
        clickCount = 1
    }

    var isLeftDown: Bool { leftDown }

    func update(thumbContact: Double,
                pinchIndex: Double,
                pinchMiddle: Double,
                fistRatio: Double,
                time: Double,
                config: GestureConfig) -> GestureOutput {
        var output = GestureOutput()

        let indexValid = pinchIndex.isFinite && pinchIndex >= 0
        let fistValid = fistRatio.isFinite

        // Fist freezes the pointer (hands get tired; freezing prevents drift while you rest).
        if config.freezeEnabled, fistValid {
            if !frozen, fistRatio > config.freezeOn {
                frozen = true
                if leftDown {
                    leftDown = false
                    leftDownFrames = 0
                    leftUpFrames = 0
                    output.events.append(.leftUp)
                }
                if rightDown {
                    rightDown = false
                    rightDownFrames = 0
                    rightUpFrames = 0
                    output.events.append(.rightUp)
                }
            } else if frozen, fistRatio < config.freezeOff {
                frozen = false
            }
        } else {
            frozen = false
        }

        guard indexValid, !frozen else {
            // Landmark confidence dropped: keep the current button state, never synthesise clicks.
            output.dragging = leftDown && config.dragEnabled
            output.pinching = leftDown
            output.frozen = frozen
            return output
        }

        // ---- Left click: finger tap (default) and/or pinch ----
        // Two independent detectors, each with its own hysteresis, OR-ed into one button. Whichever
        // is enabled decides the press; both release only when neither says "still closed", so the
        // button cannot flicker between them.
        let tapAvailable = config.tapEnabled && thumbContact.isFinite && thumbContact >= 0
        let tapClosed = tapAvailable && thumbContact < config.tapOn
        let tapOpen = !tapAvailable || thumbContact > config.tapOff
        let pinchClosed = config.pinchClickEnabled && pinchIndex < config.pinchOn
        let pinchOpen = !config.pinchClickEnabled || pinchIndex > config.pinchOff
        let wantsDown = tapClosed || pinchClosed
        let wantsUp = tapOpen && pinchOpen

        if !leftDown {
            if wantsDown {
                leftDownFrames += 1
                if leftDownFrames >= config.minDownFrames {
                    leftDown = true
                    leftDownFrames = 0
                    leftUpFrames = 0
                    if config.doubleClickEnabled, time - lastLeftUpTime <= config.doubleClickWindow {
                        clickCount = min(clickCount + 1, 3)
                    } else {
                        clickCount = 1
                    }
                    output.events.append(.leftDown(clickState: clickCount))
                }
            } else {
                leftDownFrames = 0
            }
        } else {
            if wantsUp {
                leftUpFrames += 1
                if leftUpFrames >= config.minUpFrames {
                    leftDown = false
                    leftUpFrames = 0
                    leftDownFrames = 0
                    lastLeftUpTime = time
                    output.events.append(.leftUp)
                }
            } else {
                leftUpFrames = 0
            }
        }

        // ---- Right pinch (thumb + middle), blocked while a left press is active ----
        let middleValid = pinchMiddle.isFinite && pinchMiddle >= 0
        if config.rightClickEnabled, middleValid {
            if !rightDown {
                if pinchMiddle < config.middlePinchOn, !leftDown {
                    rightDownFrames += 1
                    if rightDownFrames >= config.minDownFrames {
                        rightDown = true
                        rightDownFrames = 0
                        rightUpFrames = 0
                        output.events.append(.rightDown)
                    }
                } else {
                    rightDownFrames = 0
                }
            } else {
                if pinchMiddle > config.middlePinchOff {
                    rightUpFrames += 1
                    if rightUpFrames >= config.minUpFrames {
                        rightDown = false
                        rightUpFrames = 0
                        rightDownFrames = 0
                        output.events.append(.rightUp)
                    }
                } else {
                    rightUpFrames = 0
                }
            }
        }

        output.dragging = leftDown && config.dragEnabled
        output.pinching = leftDown
        output.frozen = frozen
        return output
    }
}

// MARK: - Scroll (two-finger pinch drag is mapped in the engine; this keeps the math testable)

struct ScrollAccumulator {
    private var remainder = 0.0
    var pixelsPerTick: Double = 12

    mutating func ticks(forPixels pixels: Double) -> Int {
        remainder += pixels
        let whole = Int(remainder / pixelsPerTick)
        remainder -= Double(whole) * pixelsPerTick
        return -whole
    }

    mutating func reset() { remainder = 0 }
}
