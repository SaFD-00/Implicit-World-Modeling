package com.monkey.collector

import android.graphics.Bitmap
import kotlin.math.abs

/**
 * Compare two low-resolution capture frames to measure visual difference.
 *
 * Ported from computer-use-preview-for-mobile's compareBitmapsKotlin(), but the
 * original exact-RGBA equality check was replaced with an integer BT.601
 * luminance comparison: each pixel is reduced to Y = (R*77 + G*150 + B*29) >> 8
 * (77/150/29 over 256 ≈ the 0.299/0.587/0.114 BT.601 luma weights) and a pixel
 * counts as changed only when its luminance differs by more than
 * LUMINANCE_THRESHOLD. This makes the settle metric ignore sub-threshold
 * rendering noise (single-LSB channel jitter) instead of tripping on it, and
 * puts the client settle decision in the same family as the server-side Stage-0
 * luminance prefilter (src/monkey_collector/pipeline/screen_matching/luminance.py).
 * Always on: there is no knob, flag, or exact-RGBA fallback path.
 */
object BitmapComparator {

    /**
     * Per-pixel BT.601 luminance delta (0–255) above which a pixel is counted as
     * changed. Value 10 matches the reference implementation and stays cross-
     * consistent with the server-side Stage-0 primitive
     * (src/monkey_collector/config.py: luminance_threshold = 10), so the client
     * settle check and the server identical-page prefilter use the same tolerance.
     */
    const val LUMINANCE_THRESHOLD = 10

    /**
     * Compare two frames and return the fraction of perceptually-changed pixels.
     *
     * Each pixel's integer BT.601 luminance is compared; a pixel counts as changed
     * only when abs(yA - yB) > LUMINANCE_THRESHOLD (strict — a delta equal to the
     * threshold does NOT count). Alpha is ignored, which is harmless because the
     * frames are opaque captures, so no separate alpha handling is needed.
     *
     * @return 0.0f (no pixel changed beyond threshold) to 1.0f (every pixel changed).
     *         Returns 1.0f if dimensions don't match (fail-safe: treat as changed).
     */
    fun compare(bitmapA: Bitmap, bitmapB: Bitmap): Float {
        if (bitmapA.width != bitmapB.width || bitmapA.height != bitmapB.height) {
            return 1.0f
        }

        val width = bitmapA.width
        val height = bitmapA.height
        val size = width * height

        val pixelsA = IntArray(size)
        val pixelsB = IntArray(size)

        bitmapA.getPixels(pixelsA, 0, width, 0, 0, width, height)
        bitmapB.getPixels(pixelsB, 0, width, 0, 0, width, height)

        var diffCount = 0
        for (i in 0 until size) {
            val a = pixelsA[i]
            val b = pixelsB[i]
            // Integer BT.601 luma: Y = (R*77 + G*150 + B*29) >> 8. Channel layout
            // matches Bitmap.getPixels()/computeFrameHash (bits 16/8/0 = R/G/B).
            val yA = (((a shr 16) and 0xFF) * 77 +
                ((a shr 8) and 0xFF) * 150 +
                (a and 0xFF) * 29) shr 8
            val yB = (((b shr 16) and 0xFF) * 77 +
                ((b shr 8) and 0xFF) * 150 +
                (b and 0xFF) * 29) shr 8
            if (abs(yA - yB) > LUMINANCE_THRESHOLD) {
                diffCount++
            }
        }

        return diffCount.toFloat() / size.toFloat()
    }

    /**
     * Compute a lightweight perceptual hash for oscillation detection.
     *
     * Divides the bitmap into an 8×8 grid and computes the average
     * luminance for each cell, producing a 64-byte fingerprint.
     * Two frames with the same hash are considered visually identical
     * for oscillation-detection purposes.
     */
    fun computeFrameHash(bitmap: Bitmap, gridSize: Int = 8): ByteArray {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        val cellW = maxOf(w / gridSize, 1)
        val cellH = maxOf(h / gridSize, 1)
        val hash = ByteArray(gridSize * gridSize)

        for (gy in 0 until gridSize) {
            for (gx in 0 until gridSize) {
                var sum = 0L
                var count = 0
                val yStart = gy * cellH
                val yEnd = minOf((gy + 1) * cellH, h)
                val xStart = gx * cellW
                val xEnd = minOf((gx + 1) * cellW, w)
                for (y in yStart until yEnd) {
                    for (x in xStart until xEnd) {
                        val pixel = pixels[y * w + x]
                        val r = (pixel shr 16) and 0xFF
                        val g = (pixel shr 8) and 0xFF
                        val b = pixel and 0xFF
                        sum += (r + g + b) / 3
                        count++
                    }
                }
                hash[gy * gridSize + gx] = if (count > 0) (sum / count).toByte() else 0
            }
        }
        return hash
    }
}
