package com.monkey.collector

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.PixelFormat
import android.graphics.Rect
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.util.Log
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Visual screen stabilization using MediaProjection low-res capture.
 *
 * Ported from computer-use-preview-for-mobile's Screen_Service.kt.
 * Captures low-resolution frames (100px wide) and compares consecutive
 * frames to detect when the screen has stopped changing (animations complete).
 */
class ScreenStabilizer(
    private val screenWidth: Int,
    private val screenHeight: Int,
    private val screenDensityDpi: Int
) {
    companion object {
        private const val TAG = "ScreenStabilizer"
        const val TARGET_WIDTH = 100
        const val STABILITY_THRESHOLD = 0.015f   // 1.5% pixel difference (stricter)
        const val FIRST_SCREEN_THRESHOLD = 0.05f // 5% — more lenient for first screen comparison
        const val REQUIRED_STABLE_FRAMES = 7     // 7 consecutive stable frames
        const val MAX_ATTEMPTS = 60              // ~19s max (1000ms initial + 60 × 300ms)
        const val CHECK_INTERVAL_MS = 300L       // 300ms between checks (higher sampling)
        const val INITIAL_WAIT_MS = 1000L        // Wait for animation to begin
        const val OSCILLATION_WINDOW = 10        // Track last 10 frame hashes
        const val OSCILLATION_MIN_REPEATS = 3    // Pattern must repeat 3+ times to be "stable"
    }

    private var mediaProjectionManager: MediaProjectionManager? = null
    private var mediaProjection: MediaProjection? = null
    private var imageReader: ImageReader? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var lastStableFrame: Bitmap? = null
    private var firstScreenFrame: Bitmap? = null
    private val isStabilizing = AtomicBoolean(false)
    @Volatile
    private var lastStabilizationTimedOut = false

    private val targetWidth: Int = TARGET_WIDTH
    private val targetHeight: Int =
        if (screenWidth > 0) (screenHeight.toFloat() / screenWidth.toFloat() * TARGET_WIDTH).toInt()
        else 178 // fallback for 1080x1920

    fun initProjection(context: Context) {
        mediaProjectionManager =
            context.getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
    }

    /**
     * Start the MediaProjection capture session with low-res ImageReader.
     * Must be called AFTER startForeground() on Android 14+.
     */
    fun startCaptureSession(resultCode: Int, data: Intent) {
        if (mediaProjectionManager == null) {
            Log.e(TAG, "MediaProjectionManager is null")
            return
        }

        // Reuse an already-live capture pipeline across sessions. Re-acquiring a
        // MediaProjection from the same consent token (resultData), or creating a
        // second VirtualDisplay on a projection whose token has timed out, throws
        // SecurityException on modern Android and KILLS the whole process — which
        // drops the standby/START handshake for every app after the first
        // (the root cause of `--apps all` collapsing after session 1). Never
        // recreate while a live projection+display already exists.
        if (mediaProjection != null && virtualDisplay != null) {
            Log.i(TAG, "Reusing existing capture session")
            return
        }

        // Release VirtualDisplay/ImageReader but keep MediaProjection alive
        stopCaptureSession()

        // Only acquire new projection if we don't have one
        if (mediaProjection == null) {
            try {
                mediaProjection = mediaProjectionManager!!.getMediaProjection(resultCode, data)
                mediaProjection?.registerCallback(object : MediaProjection.Callback() {
                    override fun onStop() {
                        Log.w(TAG, "MediaProjection stopped externally")
                        mediaProjection = null
                    }
                }, null)
            } catch (e: Exception) {
                // Consent token is single-use / timed out on modern Android.
                Log.e(TAG, "MediaProjection acquire failed (token single-use/expired): ${e.message}")
                mediaProjection = null
                return
            }
        }

        if (targetHeight <= 0) {
            Log.e(TAG, "Invalid target height: $targetHeight")
            return
        }

        imageReader = ImageReader.newInstance(
            targetWidth, targetHeight, PixelFormat.RGBA_8888, 2
        )

        try {
            virtualDisplay = mediaProjection?.createVirtualDisplay(
                "MonkeyCollector_Stabilizer",
                targetWidth,
                targetHeight,
                screenDensityDpi,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader?.surface,
                null,
                null
            )
        } catch (e: Exception) {
            // The consent token backing this MediaProjection is no longer valid
            // (single-use on modern Android). Degrade to running WITHOUT visual
            // stabilization instead of crashing the process and losing the grant.
            Log.e(TAG, "createVirtualDisplay failed (MediaProjection token invalid): ${e.message}")
            imageReader?.close()
            imageReader = null
            try { mediaProjection?.stop() } catch (_: Exception) {}
            mediaProjection = null
            virtualDisplay = null
            return
        }

        if (virtualDisplay == null) {
            Log.e(TAG, "VirtualDisplay creation failed! MediaProjection may be invalid.")
            return
        }

        Log.i(TAG, "Capture session started (${targetWidth}x${targetHeight})")
    }

    fun stopCaptureSession() {
        virtualDisplay?.release()
        virtualDisplay = null
        imageReader?.close()
        imageReader = null
        lastStableFrame?.recycle()
        lastStableFrame = null
        firstScreenFrame?.recycle()
        firstScreenFrame = null
        Log.d(TAG, "Capture session stopped")
    }

    /**
     * Full cleanup including MediaProjection.
     * Call only when the service is being destroyed.
     */
    fun release() {
        stopCaptureSession()
        mediaProjection?.stop()
        mediaProjection = null
        Log.d(TAG, "ScreenStabilizer fully released")
    }

    /**
     * Take a low-resolution comparison screenshot via MediaProjection.
     */
    fun takeComparisonScreenshot(): Bitmap? {
        if (imageReader == null) return null

        var image: Image? = null
        try {
            image = imageReader!!.acquireLatestImage() ?: return null
            return convertImageToBitmap(image)
        } catch (e: Exception) {
            Log.e(TAG, "Screenshot failed: ${e.message}")
            return null
        } finally {
            image?.close()
        }
    }

    /**
     * Wait for the screen to stabilize (stop changing).
     *
     * Waits 1000ms for animation to begin, then compares consecutive
     * low-res frames at 300ms intervals. Returns true when 7 consecutive
     * frames are within 1.5% difference.
     * Returns false on timeout (captures should still proceed).
     *
     * This is a blocking call — must be called from a worker thread.
     */
    fun waitForStable(): Boolean {
        if (!isStabilizing.compareAndSet(false, true)) {
            Log.d(TAG, "Already stabilizing, skipping")
            return true
        }

        try {
            return doWaitForStable()
        } finally {
            isStabilizing.set(false)
        }
    }

    private fun doWaitForStable(): Boolean {
        // Wait for animation to begin before starting stability checks
        Thread.sleep(INITIAL_WAIT_MS)

        var stableCount = 0
        var bitmapA: Bitmap? = null

        for (retry in 0..2) {
            bitmapA = takeComparisonScreenshot()
            if (bitmapA != null) break
            Thread.sleep(500)
        }
        if (bitmapA == null) {
            Log.w(TAG, "Initial capture failed after 3 retries, skipping stabilization")
            lastStabilizationTimedOut = false
            return true
        }

        // Ring buffer for oscillation detection
        val recentHashes = mutableListOf<ByteArray>()
        recentHashes.add(BitmapComparator.computeFrameHash(bitmapA!!))

        try {
            for (i in 1..MAX_ATTEMPTS) {
                Thread.sleep(CHECK_INTERVAL_MS)

                val bitmapB = takeComparisonScreenshot()

                val difference = if (bitmapB == null) {
                    0.0f // No new frame = screen is static
                } else {
                    BitmapComparator.compare(bitmapA!!, bitmapB)
                }

                // Track frame hash for oscillation detection
                if (bitmapB != null) {
                    val hash = BitmapComparator.computeFrameHash(bitmapB)
                    recentHashes.add(hash)
                    if (recentHashes.size > OSCILLATION_WINDOW) {
                        recentHashes.removeAt(0)
                    }

                    // Check for oscillation (e.g., cursor blink alternating 2 frames)
                    if (detectOscillation(recentHashes)) {
                        Log.d(TAG, "Oscillation detected after $i checks — treating as stable")
                        lastStableFrame?.recycle()
                        lastStableFrame = bitmapB
                        bitmapA!!.recycle()
                        lastStabilizationTimedOut = false
                        return true
                    }
                }

                if (difference < STABILITY_THRESHOLD) {
                    stableCount++
                    bitmapB?.recycle()

                    if (stableCount >= REQUIRED_STABLE_FRAMES) {
                        Log.d(TAG, "Screen stable after $i checks ($stableCount consecutive)")
                        lastStabilizationTimedOut = false
                        return true
                    }
                } else {
                    stableCount = 0
                    bitmapA!!.recycle()
                    bitmapA = bitmapB ?: takeComparisonScreenshot()
                }
            }
        } finally {
            if (bitmapA != null && !bitmapA.isRecycled) {
                bitmapA.recycle()
            }
        }

        // Timeout: save current frame as lastStableFrame to prevent
        // hasVisualChange() false positives on micro-animations
        val timeoutFrame = takeComparisonScreenshot()
        if (timeoutFrame != null) {
            lastStableFrame?.recycle()
            lastStableFrame = timeoutFrame
        }
        lastStabilizationTimedOut = true

        Log.w(TAG, "Stabilization timeout ($MAX_ATTEMPTS attempts)")
        return false
    }

    /**
     * Detect oscillating frame patterns (e.g., cursor blink alternating 2 frames).
     *
     * Checks if the last N frame hashes form a repeating cycle of period 1, 2, or 3.
     * A cycle must repeat at least OSCILLATION_MIN_REPEATS times to qualify.
     */
    private fun detectOscillation(hashes: List<ByteArray>): Boolean {
        for (period in 1..3) {
            val needed = period * OSCILLATION_MIN_REPEATS
            if (hashes.size < needed) continue

            val tail = hashes.subList(hashes.size - needed, hashes.size)
            val pattern = tail.subList(0, period)

            var matches = true
            for (j in period until needed) {
                if (!tail[j].contentEquals(pattern[j % period])) {
                    matches = false
                    break
                }
            }
            if (matches) return true
        }
        return false
    }

    /**
     * Check if the screen has visually changed since the last stable frame.
     * Also updates lastStableFrame for the next comparison.
     *
     * @return true if visual change detected, false if screen looks the same.
     */
    fun hasVisualChange(): Boolean {
        val currentFrame = takeComparisonScreenshot() ?: return true

        val previous = lastStableFrame
        if (previous == null) {
            lastStableFrame = currentFrame
            return true // First frame, always consider changed
        }

        val diff = BitmapComparator.compare(previous, currentFrame)

        previous.recycle()
        lastStableFrame = currentFrame

        // Use a more lenient threshold after stabilization timeout to
        // avoid false positives from micro-animations (cursor blink, spinners)
        val threshold = if (lastStabilizationTimedOut) FIRST_SCREEN_THRESHOLD else STABILITY_THRESHOLD
        val changed = diff > threshold
        if (!changed) {
            Log.d(TAG, "No visual change (diff=${String.format("%.4f", diff)}, threshold=${String.format("%.3f", threshold)})")
        }
        return changed
    }

    /**
     * Save the current frame as the first screen reference.
     * Called once when the very first screen capture is about to be sent.
     */
    fun saveFirstScreen() {
        val frame = takeComparisonScreenshot() ?: return
        firstScreenFrame?.recycle()
        firstScreenFrame = frame
        Log.i(TAG, "First screen saved for back-button protection")
    }

    /**
     * Check if the current screen visually matches the first screen.
     * Uses a more lenient threshold (5%) than stability checks
     * to tolerate minor dynamic content (clock, badges).
     *
     * @return true if current screen matches first screen, false otherwise.
     *         Returns false if first screen was never saved.
     */
    fun isFirstScreen(): Boolean {
        val reference = firstScreenFrame ?: return false
        val current = takeComparisonScreenshot() ?: return false
        val diff = BitmapComparator.compare(reference, current)
        current.recycle()
        val isFirst = diff < FIRST_SCREEN_THRESHOLD
        if (isFirst) {
            Log.d(TAG, "Current screen matches first screen (diff=${String.format("%.4f", diff)})")
        }
        return isFirst
    }

    /**
     * Convert an Image from ImageReader to a clean Bitmap.
     * Handles row padding artifacts from the ImageReader buffer.
     */
    private fun convertImageToBitmap(image: Image): Bitmap? {
        try {
            val planes = image.planes
            val buffer: ByteBuffer = planes[0].buffer
            val pixelStride = planes[0].pixelStride
            val rowStride = planes[0].rowStride
            val rowPadding = rowStride - pixelStride * image.width

            // Create bitmap with padding
            val rawBitmap = Bitmap.createBitmap(
                image.width + rowPadding / pixelStride,
                image.height,
                Bitmap.Config.ARGB_8888
            )
            rawBitmap.copyPixelsFromBuffer(buffer)

            // Create clean bitmap without padding
            val cleanBitmap = Bitmap.createBitmap(
                image.width,
                image.height,
                Bitmap.Config.ARGB_8888
            )
            val canvas = Canvas(cleanBitmap)
            val srcRect = Rect(0, 0, image.width, image.height)
            val dstRect = Rect(0, 0, image.width, image.height)
            canvas.drawBitmap(rawBitmap, srcRect, dstRect, null)

            rawBitmap.recycle()
            return cleanBitmap
        } catch (e: Exception) {
            Log.e(TAG, "Image→Bitmap conversion failed: ${e.message}")
            return null
        }
    }
}
