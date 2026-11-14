import Foundation
import SwiftUI

// Entry point. `--self-test` runs the measurement harness headlessly and exits, so accuracy
// claims can be checked without granting camera or accessibility permission.
if CommandLine.arguments.contains("--self-test") {
    exit(Int32(SelfTest.runAll()))
}

AeroMouseApp.main()
