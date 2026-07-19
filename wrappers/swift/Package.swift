// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "Helix",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
    ],
    products: [
        .library(name: "Helix", targets: ["Helix"]),
    ],
    targets: [
        // C header-only shim that vends helix.h to Swift
        .target(
            name: "CHelix",
            publicHeadersPath: "."
        ),
        // Swift wrapper layer
        .target(
            name: "Helix",
            dependencies: ["CHelix"],
            linkerSettings: [
                // Local dev: HELIX_LIB_DIR env var or the build/ directory
                // In xcframework distribution, the framework handles linking.
                .linkedLibrary("helix"),
            ]
        ),
        .testTarget(
            name: "HelixTests",
            dependencies: ["Helix"]
        ),
    ]
)
