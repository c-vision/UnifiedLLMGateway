package main

import (
	"bytes"
	"image"
	"image/color"
	"image/draw"
	"image/png"
)

// buildIcon renders a block letter "U" (for "Unified") in solid black, as
// a macOS "template" image (opaque shape + alpha channel; macOS recolors
// it for light/dark menu bars, same as other system tray icons). Status
// is conveyed by the 🟢/🟡/🔴 text in the menu items instead of the tray
// icon's color, so it doesn't change based on gateway state.
func buildIcon() []byte {
	const size = 22
	const strokeWidth = 3
	const letterWidth = 12
	const letterHeight = 14
	left := (size - letterWidth) / 2
	top := (size - letterHeight) / 2
	black := color.RGBA{0, 0, 0, 255}

	img := image.NewRGBA(image.Rect(0, 0, size, size))
	fill := func(x0, y0, x1, y1 int) {
		draw.Draw(img, image.Rect(x0, y0, x1, y1), &image.Uniform{black}, image.Point{}, draw.Src)
	}
	fill(left, top, left+strokeWidth, top+letterHeight)                          // left stroke
	fill(left+letterWidth-strokeWidth, top, left+letterWidth, top+letterHeight)  // right stroke
	fill(left, top+letterHeight-strokeWidth, left+letterWidth, top+letterHeight) // bottom stroke

	var buf bytes.Buffer
	_ = png.Encode(&buf, img)
	return buf.Bytes()
}
