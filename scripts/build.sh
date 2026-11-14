#!/usr/bin/env bash
# Builds AeroMouse.app. Uses swiftc directly because SwiftPM's build system needs a full Xcode
# installation, while this project only needs the Command Line Tools.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

APP="build/AeroMouse.app"
BIN="$APP/Contents/MacOS/AeroMouse"

echo "Compiling…"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# shellcheck disable=SC2046
swiftc -O -module-name AeroMouse -o "$BIN" $(find Sources/AeroMouse -name '*.swift' | sort)

cp Resources/Info.plist "$APP/Contents/Info.plist"

echo "Signing (ad-hoc)…"
codesign --force --sign - "$APP"

echo "Built $APP"
echo "Open it with:  open \"$ROOT/$APP\""
