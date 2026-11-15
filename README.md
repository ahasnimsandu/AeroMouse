# AeroMouse

Control the mouse pointer with your hand. A camera watches your hand, a calibrated model turns it
into pointer motion, and pinching thumb + index clicks. Everything runs on-device — the camera
feed never leaves your Mac.

macOS 14+ · Apple Silicon or Intel · no dependencies beyond the Xcode Command Line Tools.

## Install

Download `AeroMouse.dmg` from [Releases](../../releases), open it, and drag **AeroMouse** into
**Applications**.

The build is signed ad-hoc rather than notarized, so the first launch is blocked by Gatekeeper.
Right-click the app → **Open** → **Open**, or clear the quarantine flag once:

```bash
xattr -dr com.apple.quarantine /Applications/AeroMouse.app
```

## Build from source

```bash
./scripts/build.sh
open build/AeroMouse.app
```

It compiles with `swiftc` directly, so a full Xcode installation is not needed. Run the app from
the `.app` bundle — camera permission is tied to the bundle identity.

To package a disk image:

```bash
./scripts/make-dmg.sh          # → build/AeroMouse.dmg
```

## First launch

1. macOS asks for **Camera** access. Allow it.
2. Open the menu bar panel → **Permissions** → **Open Settings** for **Accessibility** (needed to
   move the cursor and synthesize clicks). Tick AeroMouse, then quit and reopen the app.
3. **Calibrate…** — follow the dots. Pinch thumb + index on each one.

Use `--self-test` at any time to check the maths without granting anything:

```bash
build/AeroMouse.app/Contents/MacOS/AeroMouse --self-test
```

## Gestures

| Gesture | Action |
| --- | --- |
| Tap: thumb pad against the side of the index finger | Left click |
| Keep the contact and move | Drag |
| Two quick taps | Double click |
| Thumb + middle pinch | Right click |
| Index + middle held together, hand moved up/down | Scroll |
| Closed fist | Freeze the pointer |
| Open hand | Move the pointer |

Scroll is a separate pose rather than a button: with the thumb clear and the two fingers together the
hand sends wheel events, so the pointer stays exactly where it is instead of sliding down the page
while you scroll.

The click gesture is the Apple Watch double-tap geometry rather than a pinch: the gap is measured
from the thumb tip to the *edge* of the index finger (the segment from the first to the last
knuckle), not to the fingertip. A tap anywhere along that edge registers, the reading does not
change when the finger curls, and the measurement is divided by hand size so it does not depend on
distance to the camera. Classic pinch-to-click is still available as a toggle.

A fixed threshold cannot work for a tap — one thumb rests at 0.10 from the finger, another at 0.30 —
and guessing is what made ordinary hand movement click. The recogniser therefore learns the **resting
gap** and requires a real drop below it, while an absolute ceiling stops a mis-learned baseline from
turning rest into clicks. Press *Learn my tap…*, relax your hand, tap three times, and it places both
thresholds from your own readings. The panel shows the resting gap, the level it fires below and the
level it releases above, live.

Every threshold has a hysteresis gap and a frame debounce, so a hand resting exactly on the
boundary cannot produce flickering clicks. Losing the hand entirely releases a held button instead
of leaving the mouse stuck down.

## How the accuracy works

Calibration collects two kinds of samples: a pinch at a target (a deliberate, precise point) and a
settled hover near it (cheap extra coverage). Then:

1. **Four mappings are fitted** — projective homography, quadratic and cubic ridge regression, and
   pose-conditioned ridge regression (position × depth/roll/spread interaction terms, which is what
   captures moving your hand closer to the camera). Every one is solved by Householder QR least
   squares with damping rows appended, because solving through the normal equations squares the
   condition number and destroys the cubic fit.
2. **Grouped cross-validation** scores each candidate, holding out whole target cells so a model
   can never score well by memorizing a neighbouring point.
3. **The good ones are blended** by inverse error with clearly worse fits dropped, then a second,
   smaller quadratic stage is fitted on the blend's residual. It is a cascade: each stage only
   learns what the previous one missed.
4. **A local residual stage** re-fits a small weighted quadratic around each query point using only
   the nearby samples. Global fits have to average away curvature between the dots — and those
   gaps are where you actually use the pointer. Measured against a synthetic truth that bends
   mid-cell, this cuts 22.1 px to 12.9 px, and it fades out where samples are sparse.
5. **A neural residual network** (2 hidden layers, trained from scratch with Adam on your own
   samples) learns what geometry cannot express — depth changes, wrist roll, finger spread. It is
   kept only when cross-validation says it helps, and its influence fades out outside the
   calibrated area so it cannot extrapolate wildly. It is skipped when the local stage wins, so the
   two never double-correct the same residual.

The panel reports median and p95 error, before and after, on held-out cells. On the synthetic
harness the full stack takes **8.4 px → 1.8 px median (4.6×)** with a 4.2 px p95.

## Low light and weak cameras

Vision's hand-pose network already runs on the Neural Engine, so the remaining wins are in the
signal rather than the model:

- **Sharp landmarks first.** The camera captures at 1280×720 by default instead of 640×480: twice
  the pixels halves the landmark error, and that error is multiplied again on the way to the screen.
  Turn *HD capture* off if you would rather have a constant 60 fps.
- **Longer exposure.** In dim conditions the camera drops to 30 fps, which lets it integrate light
  for twice as long. That removes sensor noise at the source instead of denoising it afterwards.
- **GPU frame conditioning** (Core Image on Metal): a tone-curve shadow lift, a light unsharp pass
  to restore the edge structure noise smears, and temporal averaging that grows as the light falls.
  Bright frames take the untouched path, so nothing is paid in good light.
- **All 21 landmarks are filtered**, not just the pointer anchor. Confident joints are smoothed;
  joints below 0.30 confidence are held for up to 220 ms rather than followed, which stops a
  mis-detected finger from producing a phantom pinch.
- **Quality gating.** A per-frame score (landmark confidence + light) widens the deadband when
  tracking is poor and down-weights noisy samples during calibration, so a dark frame cannot skew
  the fitted model.

Both behaviours are switches in the panel if you would rather keep a constant 60 fps.

## Accuracy chart

After calibrating, the Accuracy card draws:

- an **error map** — one ring per calibration cell, coloured by median error, with an arrow showing
  the systematic bias direction (magnified 6×) so you can see *where* the model leans;
- an **error histogram** of every held-out sample, before versus after the stack;
- **percentile bars** (p50/p75/p90/p95/max) in pixels, before versus after.

## Sub-pixel finger tracking

Vision gives an absolute landmark every frame, but its noise is a pixel or two in image space, and
the screen mapping multiplies that into several screen pixels. Motion between two consecutive frames
is far easier to measure precisely than position, so a small patch of the previous frame is matched
against the current one, the match is refined to a fraction of a pixel by fitting a parabola through
the minimum, and that patch is carried forward as the motion model. The landmark stays the absolute
reference and corrects any drift, so this behaves like a Kalman filter whose process model is
measured image evidence instead of an assumption about velocity.

Every failure mode falls back to the landmark: weak texture, a failed or border-of-window match, a
dark frame, an unrecognised pixel format, or a disagreement large enough to be a jump rather than
motion. On a synthetic frame shifted by a deliberately fractional 6.4 px, the recovered motion is
accurate to **0.05 px**. Toggle it off under *Low light & GPU* if you ever want the plain landmark.

## Feel presets

The panel has three one-click presets — **Precise**, **Balanced** and **Snappy** — that move
smoothing, prediction, precision gain and deadband together. Tuning those four independently is what
usually makes a pointer feel subtly wrong; a preset keeps them consistent and the sliders stay
available underneath for anyone who wants to go further.

## Smoothing

- One Euro adaptive filter — low cutoff when still, high cutoff when moving fast
- Constant-velocity prediction, gated by speed so it never overshoots at rest
- Precision gain: slow movement is attenuated so small targets are reachable
- Outlier gate: one bad frame holds the cursor instead of teleporting it
- Deadband: no one-pixel shimmer while your hand is still
- Click lock: the cursor does not twitch when you pinch

## Gaze assist

A webcam has no depth sensor, so gaze cannot steer a pointer on its own — but it is an excellent
absolute reference, and that is what it is used for.

The model takes both pupils' positions inside their eye openings, head yaw, pitch and roll, the head's
position in frame and its apparent size (a distance cue), plus quadratic terms, and fits them to the
screen with ridge regression. Eye opening is measured too: blink frames are rejected instead of
poisoning the fit. Accuracy is reported by grouped cross-validation over the calibration targets, so
the number in the panel is what you would actually get at a spot you did not calibrate on, and a
model that cannot reach a useful threshold is discarded rather than applied weakly.

At runtime the correction is gated three ways: gain follows the model's measured accuracy, the clamp
scales with the same figure (a shaky model can only nudge, a good one can pull up to 9% of the
screen), and it fades in only as the hand settles — which is exactly when precision matters and when
a slow pull cannot smear intentional motion. Strength is a slider from off to full.

Honest limit: on a webcam this is tens of pixels, not a few. It removes drift and lands you in the
right neighbourhood; it does not replace the hand for fine pointing.

## Privacy

Camera frames are processed in memory and discarded. There is no network code in this app: no
analytics, no telemetry, no accounts. The only files it writes are the model and settings below.

## Files

- Model: `~/Library/Application Support/AeroMouse/model.json`
- Settings: `~/Library/Application Support/AeroMouse/settings.json`

Set `AEROMOUSE_DEBUG=1` to print per-stage cross-validation error while calibrating.

## Limitations

- One hand at a time, and the hand should stay roughly arm's length from the camera.
- Accuracy degrades at the screen edges beyond the calibrated area; recalibrate if you move the
  camera or change chairs.
- No trackpad-style gestures (scroll, swipe) yet.

## License

MIT — see [LICENSE](LICENSE).
