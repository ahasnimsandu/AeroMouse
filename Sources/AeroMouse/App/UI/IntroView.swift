import SwiftUI
import AppKit

final class IntroState: ObservableObject {
    @Published var page: Int = 0
}

struct IntroView: View {
    @ObservedObject var app: AppState
    var onDismiss: () -> Void
    @StateObject private var state = IntroState()

    private var page: Int { state.page }
    private func setPage(_ value: Int) { state.page = value }

    private let pages: [IntroPage] = [
        IntroPage(icon: "hand.raised.fill",
                  title: "Point with your hand",
                  subtitle: "AeroMouse turns your webcam into a precise mouse. No extra hardware, no dongle — just your hand in front of the camera."),
        IntroPage(icon: "hand.tap.fill",
                  title: "Tap to click",
                  subtitle: "Tap your thumb against the side of your index finger, like Apple Watch double tap. A quick tap clicks; keep them together and move to drag."),
        IntroPage(icon: "hand.point.up.left.fill",
                  title: "Right click",
                  subtitle: "Tap your thumb against your middle finger instead. Same gesture, different finger."),
        IntroPage(icon: "arrow.up.arrow.down",
                  title: "Scroll",
                  subtitle: "Hold your index and middle fingers together, keep the thumb away, then move your hand up or down. The cursor stays where it is."),
        IntroPage(icon: "arrow.up.right.and.arrow.down.left",
                  title: "Reach every corner",
                  subtitle: "Move closer to the camera for full control. If your hand leaves the frame, AeroMouse glides to the edge and picks up where you left off."),
        IntroPage(icon: "checkmark.circle.fill",
                  title: "You're ready",
                  subtitle: "Allow camera and Accessibility access, calibrate on your screen, then teach it your tap. You can reopen this guide from the menu bar."),
    ]

    var body: some View {
        ZStack {
            LinearGradient(colors: [Color(red: 0.07, green: 0.10, blue: 0.22),
                                    Color(red: 0.16, green: 0.10, blue: 0.30)],
                           startPoint: .topLeading, endPoint: .bottomTrailing)
                .ignoresSafeArea()

            VStack(spacing: 0) {
                HStack {
                    Spacer()
                    Button("Skip") { finish() }
                        .buttonStyle(.plain)
                        .foregroundStyle(.white.opacity(0.7))
                        .padding(.trailing, 24)
                        .padding(.top, 18)
                }

                Spacer(minLength: 12)

                pageView(pages[page])
                    .frame(height: 360)
                    .id(page)
                    .transition(.opacity.combined(with: .move(edge: .trailing)))

                HStack(spacing: 8) {
                    ForEach(0..<pages.count, id: \.self) { i in
                        Capsule()
                            .fill(i == page ? Color.white : Color.white.opacity(0.3))
                            .frame(width: i == page ? 22 : 7, height: 7)
                            .animation(.spring(response: 0.35, dampingFraction: 0.8), value: page)
                    }
                }
                .padding(.top, 8)

                HStack(spacing: 12) {
                    if page > 0 {
                        Button {
                            withAnimation(.easeInOut(duration: 0.25)) { setPage(page - 1) }
                        } label: {
                            Image(systemName: "chevron.left")
                                .font(.title3.weight(.semibold))
                                .foregroundStyle(.white)
                                .frame(width: 44, height: 44)
                                .background(Circle().fill(Color.white.opacity(0.12)))
                        }
                        .buttonStyle(.plain)
                    } else {
                        Spacer().frame(width: 44)
                    }

                    Spacer()

                    Button(page == pages.count - 1 ? "Get started" : "Continue") {
                        if page == pages.count - 1 { finish() }
                        else { withAnimation(.easeInOut(duration: 0.25)) { setPage(page + 1) } }
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .tint(Color.white.opacity(0.18))
                    .foregroundStyle(.white)
                }
                .padding(.horizontal, 32)
                .padding(.bottom, 26)
                .padding(.top, 18)
            }
        }
        .frame(width: 720, height: 500)
    }

    private func pageView(_ p: IntroPage) -> some View {
        VStack(spacing: 20) {
            ZStack {
                Circle()
                    .fill(Color.white.opacity(0.10))
                    .frame(width: 120, height: 120)
                Image(systemName: p.icon)
                    .font(.system(size: 52, weight: .semibold))
                    .foregroundStyle(.white)
                    .symbolRenderingMode(.hierarchical)
            }
            .padding(.top, 8)

            Text(p.title)
                .font(.system(size: 28, weight: .bold, design: .rounded))
                .foregroundStyle(.white)
                .multilineTextAlignment(.center)

            Text(p.subtitle)
                .font(.body)
                .foregroundStyle(.white.opacity(0.78))
                .multilineTextAlignment(.center)
                .frame(maxWidth: 520)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 40)
    }

    private func finish() {
        UserDefaults.standard.set(true, forKey: "hasSeenIntro")
        onDismiss()
    }
}

private struct IntroPage {
    let icon: String
    let title: String
    let subtitle: String
}

final class IntroWindowController {
    static let shared = IntroWindowController()
    private var window: NSWindow?

    func show(app: AppState) {
        if let window {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }
        let view = IntroView(app: app) { [weak self] in self?.close() }
        let hosting = NSHostingController(rootView: view)
        let w = NSWindow(contentViewController: hosting)
        w.styleMask = [.titled, .closable, .fullSizeContentView]
        w.titlebarAppearsTransparent = true
        w.titleVisibility = .hidden
        w.isMovableByWindowBackground = true
        w.setContentSize(NSSize(width: 720, height: 500))
        w.center()
        w.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        window = w
    }

    func close() {
        window?.close()
        window = nil
    }
}