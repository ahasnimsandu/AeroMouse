#!/usr/bin/env bash
# Packages build/AeroMouse.app into build/AeroMouse.dmg with an Applications shortcut.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

APP="build/AeroMouse.app"
DMG="build/AeroMouse.dmg"
STAGE="build/dmg-stage"

[ -d "$APP" ] || { echo "No $APP — run ./scripts/build.sh first." >&2; exit 1; }

rm -rf "$STAGE" "$DMG"
mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"

hdiutil create -quiet -volname "AeroMouse" -srcfolder "$STAGE" -format UDZO -ov "$DMG"
rm -rf "$STAGE"

echo "Built $DMG"
