// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "AeroMouse",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "AeroMouse",
            path: "Sources/AeroMouse"
        )
    ]
)
