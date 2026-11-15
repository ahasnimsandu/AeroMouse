import Cocoa
import Foundation

let iconsetDir = "Resources/AppIcon.iconset"
let fm = FileManager.default
try? fm.removeItem(atPath: iconsetDir)
try fm.createDirectory(atPath: iconsetDir, withIntermediateDirectories: true)

let entries: [(String, Int)] = [
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
]

func drawIcon(size: Int) -> NSImage {
    let s = CGFloat(size)
    let image = NSImage(size: NSSize(width: s, height: s))
    image.lockFocus()
    guard let ctx = NSGraphicsContext.current?.cgContext else {
        image.unlockFocus()
        return image
    }

    let corner = s * 0.225
    let rect = NSRect(x: 0, y: 0, width: s, height: s)
    let path = NSBezierPath(roundedRect: rect, xRadius: corner, yRadius: corner)
    path.addClip()

    // Gradient background
    let colors = [NSColor(srgbRed: 0.18, green: 0.32, blue: 0.95, alpha: 1),
                  NSColor(srgbRed: 0.55, green: 0.22, blue: 0.92, alpha: 1)] as CFArray
    let space = CGColorSpaceCreateDeviceRGB()
    if let gradient = CGGradient(colorsSpace: space, colors: colors, locations: [0, 1]) {
        ctx.drawLinearGradient(gradient, start: CGPoint(x: 0, y: s), end: CGPoint(x: s, y: 0), options: [])
    }

    // Subtle highlight
    ctx.saveGState()
    let highlight = NSBezierPath(ovalIn: NSRect(x: -s*0.1, y: s*0.55, width: s*1.2, height: s*0.6))
    highlight.addClip()
    let hColors = [NSColor.white.withAlphaComponent(0.18).cgColor,
                   NSColor.white.withAlphaComponent(0.0).cgColor] as CFArray
    if let hg = CGGradient(colorsSpace: space, colors: hColors, locations: [0, 1]) {
        ctx.drawLinearGradient(hg, start: CGPoint(x: s/2, y: s), end: CGPoint(x: s/2, y: s*0.5), options: [])
    }
    ctx.restoreGState()

    // SF Symbol
    let symbolSize = s * 0.52
    let config = NSImage.SymbolConfiguration(pointSize: symbolSize, weight: .semibold)
    if let symbol = NSImage(systemSymbolName: "hand.point.up.left.fill", accessibilityDescription: nil)?.withSymbolConfiguration(config) {
        let symSize = symbol.size
        let x = (s - symSize.width) / 2
        let y = (s - symSize.height) / 2 + s * 0.02
        // Shadow
        ctx.saveGState()
        ctx.setShadow(offset: CGSize(width: 0, height: -s*0.02), blur: s*0.06,
                      color: NSColor.black.withAlphaComponent(0.35).cgColor)
        symbol.draw(in: NSRect(x: x, y: y, width: symSize.width, height: symSize.height))
        ctx.restoreGState()
    }

    image.unlockFocus()
    return image
}

func writePNG(_ image: NSImage, to path: String) {
    guard let tiff = image.tiffRepresentation,
          let rep = NSBitmapImageRep(data: tiff),
          let png = rep.representation(using: .png, properties: [:]) else { return }
    try? png.write(to: URL(fileURLWithPath: path))
}

for (name, size) in entries {
    let img = drawIcon(size: size)
    writePNG(img, to: "\(iconsetDir)/\(name)")
    print("wrote \(name)")
}
print("done")