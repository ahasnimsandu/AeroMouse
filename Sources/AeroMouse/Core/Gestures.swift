import Foundation

// MARK: - Gesture configuration

struct GestureConfig: Codable {
    /// Tap: the thumb pad meets the edge of the index finger, like Apple Watch double tap.
    /// Measured in hand units, so it does not depend on how big the hand or the frame is.
    var tapEnabled: Bool = true
    /// A fixed threshold cannot work: a relaxed thumb sits at 0.10 for one person and 0.30 for
    /// another, so the detector learns the resting gap and requires a real drop below it. The
    /// absolute value stays as a ceiling, so a bad baseline can never make it trigger on rest.
    var tapAdaptiveThreshold: Bool = true
    var tapOn: Double = 0.075         // absolute ceiling: never click above this gap
    var tapOff: Double = 0.125
    var tapBaseline: Double = 0.22    // learned resting gap, overwritten by the trainer
    var tapDrop: Double = 0.05        // how far below the resting gap a tap must go
    var pinchClickEnabled: Bool = false
    var pinchOn: Double = 0.42        // thumb-index distance / hand scale to close
    var pinchOff: Double = 0.62       // ...and to open again (hysteresis gap kills flicker)

    /// Scroll: index and middle fingers held together, thumb clear, hand moved up or down.
    var scrollEnabled: Bool = true
    var scrollMinGap: Double = 0.40   // gap that enters scroll mode
    var scrollMaxGap: Double = 0.55   // gap that leaves it again
    var scrollThumbClear: Double = 0.60  // thumb must be this far from the index, or it is a tap
    var scrollSpeed: Double = 2.4     // pointer pixels of hand travel per wheel step
    var scrollDeadzone: Double = 0.6  // px of hand travel ignored, to kill tremor
    var middlePinchOn: Double = 0.40
    var middlePinchOff: Double = 0.60
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
    var scrolling: Bool = false
    var frozen: Bool = false
    /// Live detector state, so the panel can show exactly what the recogniser is comparing against.
    var tapOnLevel: Double = 0
    var tapOffLevel: Double = 0
    var contactBaseline: Double = 0
    var tapClosed: Bool = false
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

    // Adaptive tap baseline state.
    private var contactBaseline: Double = 0
    private var baselineFrames = 0
    private var lastUpdateTime: Double = -1
    private var scrolling = false

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
        contactBaseline = 0
        baselineFrames = 0
        lastUpdateTime = -1
        scrolling = false
    }

    /// Resting gap the detector is currently assuming, for the panel and the self-test.
    var learningBaseline: Double { contactBaseline }

    var isLeftDown: Bool { leftDown }

    func update(thumbContact: Double,
                pinchIndex: Double,
                pinchMiddle: Double,
                indexMiddleGap: Double,
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

        // ---- Adaptive tap baseline ----
        // Learn the gap a *resting* hand shows. It follows the resting value up quickly (the hand
        // changes pose, moves closer) and down very slowly, so a tap cannot drag the baseline with
        // it and teach the detector to ignore the next tap.
        let dt = lastUpdateTime < 0 ? 1.0 / 60.0 : max(min(time - lastUpdateTime, 0.2), 1.0 / 240.0)
        lastUpdateTime = time
        if thumbContact.isFinite, thumbContact > 0, !leftDown {
            if contactBaseline <= 0 || baselineFrames < 40 {
                contactBaseline = contactBaseline <= 0
                    ? thumbContact
                    : contactBaseline + (thumbContact - contactBaseline) * 0.2
                baselineFrames += 1
            } else {
                let alpha = thumbContact > contactBaseline ? 2.5 : 0.06
                contactBaseline += (thumbContact - contactBaseline) * min(alpha * dt, 1)
            }
        }

        let baseline = contactBaseline > 0 ? contactBaseline : config.tapBaseline
        // Until the resting gap has been observed for a moment, behave exactly like the plain
        // absolute threshold: a hand that starts out mid-tap must still be able to click.
        let adaptive = config.tapAdaptiveThreshold && baselineFrames >= 40
        // Fire only below BOTH the learned drop and the absolute ceiling: the adaptive part handles
        // anatomy, the ceiling stops a runaway baseline from turning rest into clicks.
        let onLevel = adaptive
            ? min(config.tapOn, max(0.02, baseline - config.tapDrop))
            : config.tapOn
        // Release must always sit below the *resting* gap, otherwise a hand whose thumb rests at
        // 0.11 against a release threshold of 0.125 would hold the button down forever — a stuck
        // drag. It also has to stay above the press level, or the button would chatter.
        let offLevel = max(onLevel + 0.015,
                           min(config.tapOff, baseline - config.tapDrop * 0.35))

        guard indexValid, !frozen else {
            // Landmark confidence dropped: keep the current button state, never synthesise clicks.
            output.dragging = leftDown && config.dragEnabled
            output.pinching = leftDown
            output.frozen = frozen
            output.tapOnLevel = onLevel
            output.tapOffLevel = offLevel
            output.contactBaseline = baseline
            return output
        }

        // ---- Left click: finger tap (default) and/or pinch ----
        // Two independent detectors, each with its own hysteresis, OR-ed into one button. Whichever
        // is enabled decides the press; both release only when neither says "still closed", so the
        // button cannot flicker between them.
        let tapAvailable = config.tapEnabled && thumbContact.isFinite && thumbContact >= 0
        let tapClosed = tapAvailable && thumbContact < onLevel
        let tapOpen = !tapAvailable || thumbContact > offLevel
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
        // A right click is a deliberate middle-finger pinch: never fire it while the thumb is
        // anywhere near the index finger, or an ordinary click gets misread as a right click.
        let leftAttempting = tapAvailable && thumbContact.isFinite && thumbContact < offLevel
        if leftAttempting, rightDown {
            rightDown = false
            rightDownFrames = 0
            rightUpFrames = 0
            output.events.append(.rightUp)
        }
        if config.rightClickEnabled, middleValid, !leftAttempting {
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

        // ---- Scroll: index and middle fingers held together, thumb out of the way ----
        let gapValid = indexMiddleGap.isFinite && indexMiddleGap >= 0
        if config.scrollEnabled, gapValid {
            let limit = scrolling ? config.scrollMaxGap : config.scrollMinGap
            scrolling = indexMiddleGap < limit
                && pinchIndex > config.scrollThumbClear
                && !leftDown && !rightDown
        } else {
            scrolling = false
        }

        output.dragging = leftDown && config.dragEnabled
        output.pinching = leftDown
        output.scrolling = scrolling
        output.frozen = frozen
        output.tapOnLevel = onLevel
        output.tapOffLevel = offLevel
        output.contactBaseline = baseline
        output.tapClosed = tapClosed
        return output
    }
}

// MARK: - Tap trainer

/// Learns a user's own tap from a few seconds of contact readings: the resting gap (high quantile)
/// and the tapped gap (low quantile). Thresholds are placed between the two, so the detector ends up
/// tuned to this hand, this camera angle and this distance instead of to a guess.
enum TapTrainer {
    struct Suggestion: Equatable {
        var onLevel: Double
        var offLevel: Double
        var baseline: Double
        var resting: Double
        var tapped: Double
    }

    static func suggestion(from contacts: [Double]) -> Suggestion? {
        let usable = contacts.filter { $0.isFinite && $0 > 0 }.sorted()
        guard usable.count >= 60 else { return nil }

        func quantile(_ q: Double) -> Double {
            let index = Int((Double(usable.count - 1) * q).rounded())
            return usable[min(max(index, 0), usable.count - 1)]
        }

        let resting = quantile(0.85)
        let tapped = quantile(0.06)
        // A hand that never actually tapped gives no usable separation: leave the settings alone
        // rather than silently teaching the detector to fire on anything.
        guard resting - tapped > 0.045 else { return nil }

        let span = resting - tapped
        let onLevel = min(0.14, tapped + span * 0.30)
        let offLevel = max(onLevel + 0.02, tapped + span * 0.60)
        return Suggestion(onLevel: onLevel, offLevel: offLevel,
                          baseline: resting, resting: resting, tapped: tapped)
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
