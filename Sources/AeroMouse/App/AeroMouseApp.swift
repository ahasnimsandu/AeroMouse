import SwiftUI
import AppKit

struct AeroMouseApp: App {
    @StateObject private var app = AppState()
    @Environment(\.openWindow) private var openWindow

    var body: some Scene {
        MenuBarExtra {
            Button(app.isTracking ? "Pause tracking" : "Start tracking") {
                app.toggleTracking()
            }
            Button("Calibrate…") {
                app.startCalibration()
            }
            Button(modelSummary) {
                NSApp.activate(ignoringOtherApps: true)
                openWindow(id: "control")
            }
            Divider()
            Button("Open control panel") {
                NSApp.activate(ignoringOtherApps: true)
                openWindow(id: "control")
            }
            Divider()
            Button("Quit AeroMouse") {
                NSApp.terminate(nil)
            }
        } label: {
            Image(systemName: app.isTracking ? "hand.point.up.left.fill" : "hand.point.up.left")
        }

        Window("AeroMouse", id: "control") {
            ControlPanelView()
                .environmentObject(app)
        }
        .defaultSize(width: 860, height: 780)
        .windowResizability(.contentMinSize)
    }

    private var modelSummary: String {
        guard let stats = app.model?.stats else { return "No calibration yet" }
        return String(format: "Calibrated — %.1f px median", stats.medianAfter)
    }
}
