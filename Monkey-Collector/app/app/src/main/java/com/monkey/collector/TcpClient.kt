package com.monkey.collector

import android.graphics.Bitmap
import android.util.Log
import java.io.BufferedReader
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.io.IOException
import java.io.InputStreamReader
import java.net.Socket
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import org.json.JSONObject

class TcpClient(
    private val serverIp: String,
    private val serverPort: Int
) {
    companion object {
        private const val TAG = "TcpClient"
        private const val MAX_RETRIES = 3
        private const val RETRY_DELAY_MS = 2000L
    }

    private var socket: Socket? = null
    private var dos: DataOutputStream? = null
    private val writeLock = Any()
    private var readerThread: Thread? = null
    private var onSessionEnd: (() -> Unit)? = null
    private var onStart: ((String) -> Unit)? = null
    private var onCaptureRequest: (() -> Unit)? = null

    @Volatile
    private var connected = false

    /**
     * Frame-tracking state for the CAPTURE-poke decision, owned by this class so
     * it can be updated *inside* [sendFrame]'s [writeLock] section.
     *
     * Both fields are written only while the lock is held, so their write order
     * is the wire order. They are read lock-free (volatile) on purpose: the
     * [onCaptureRequest] callback runs on the reader thread, and making it wait
     * for [writeLock] would stall SESSION_END/CAPTURE handling behind a large
     * in-flight JPEG. The reader can therefore see state that is at most one
     * in-progress send stale — and a stale read only ever costs a duplicate
     * frame, never a mispaired one.
     */
    @Volatile
    var lastSentXmlHash: String? = null
        private set

    @Volatile
    var lastFrameSentAt: Long = 0L
        private set

    /** Clear frame-tracking state at the start of a collection session. */
    fun resetFrameTracking() {
        lastSentXmlHash = null
        lastFrameSentAt = 0L
    }

    fun setOnSessionEnd(callback: () -> Unit) {
        onSessionEnd = callback
    }

    fun setOnStart(callback: (String) -> Unit) {
        onStart = callback
    }

    /**
     * Register the handler for server CAPTURE pokes.
     *
     * The callback runs on the reader thread, so it MUST return immediately —
     * any screen capture work belongs on a thread the callback spawns.
     */
    fun setOnCaptureRequest(callback: () -> Unit) {
        onCaptureRequest = callback
    }

    fun connect(): Boolean {
        for (attempt in 1..MAX_RETRIES) {
            try {
                socket = Socket(serverIp, serverPort)
                dos = DataOutputStream(socket!!.getOutputStream())
                connected = true
                startReaderThread()
                Log.i(TAG, "Connected to $serverIp:$serverPort (attempt $attempt)")
                return true
            } catch (e: IOException) {
                Log.e(TAG, "Connection attempt $attempt/$MAX_RETRIES failed: ${e.message}")
                if (attempt < MAX_RETRIES) {
                    try {
                        Thread.sleep(RETRY_DELAY_MS)
                    } catch (_: InterruptedException) {
                        break
                    }
                }
            }
        }
        Log.e(TAG, "Failed to connect after $MAX_RETRIES attempts")
        return false
    }

    /**
     * Start a background thread that reads control signals from the server.
     *
     * The server sends \r\n-delimited JSON messages. Supported types:
     * - {"type": "START", "package": "<pkg>"} — server requests the app to
     *   begin collecting the given package.
     * - {"type": "SESSION_END"} — server requests the app to stop collection.
     * - {"type": "CAPTURE"} — server heard nothing back after an action and
     *   asks for the current screen state; the client replies with N, E or S+X
     *   without waiting for an accessibility event.
     */
    private fun startReaderThread() {
        readerThread = Thread {
            try {
                val reader = BufferedReader(
                    InputStreamReader(socket!!.getInputStream(), StandardCharsets.UTF_8)
                )
                while (connected) {
                    val line = reader.readLine() ?: break  // null = connection closed
                    try {
                        val json = JSONObject(line.trim())
                        when (json.optString("type")) {
                            "START" -> {
                                val pkg = json.optString("package", "")
                                if (pkg.isEmpty()) {
                                    Log.w(TAG, "START message missing package field: $line")
                                } else {
                                    Log.i(TAG, "Received START from server: $pkg")
                                    onStart?.invoke(pkg)
                                }
                            }
                            "SESSION_END" -> {
                                Log.i(TAG, "Received SESSION_END from server")
                                onSessionEnd?.invoke()
                            }
                            "CAPTURE" -> {
                                Log.d(TAG, "Received CAPTURE from server")
                                onCaptureRequest?.invoke()
                            }
                            else -> Log.d(TAG, "Unknown server message: $line")
                        }
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to parse server message: $line")
                    }
                }
            } catch (e: IOException) {
                if (connected) {
                    Log.w(TAG, "Reader thread error: ${e.message}")
                }
            }
            Log.d(TAG, "Reader thread exited")
        }.apply {
            isDaemon = true
            name = "TcpClient-Reader"
            start()
        }
    }

    fun disconnect() {
        connected = false
        readerThread?.interrupt()
        readerThread = null
        synchronized(writeLock) {
            try {
                dos?.close()
                socket?.close()
            } catch (e: IOException) {
                Log.e(TAG, "Disconnect error: ${e.message}")
            } finally {
                dos = null
                socket = null
            }
        }
    }

    fun isConnected(): Boolean = connected

    /** Per-frame outcome: the S and X halves can succeed independently. */
    data class FrameSendResult(val screenshotSent: Boolean, val xmlSent: Boolean)

    /**
     * Send one capture as an atomic screenshot + XML pair.
     *
     * Both frames are written inside a *single* [writeLock] section on purpose.
     * Captures can overlap: the accessibility-event path spawns a capture thread
     * on every debounce-passing event without checking whether one is already in
     * flight, and the server CAPTURE poke can spawn one too. If each frame took
     * the lock separately, two overlapping captures A and B could interleave on
     * the wire as S(A) S(B) X(B) X(A) — the server pairs the newest screenshot
     * with the next XML, so it would attach A's pixels to B's tree. Holding the
     * lock across both frames guarantees the server sees complete, ordered pairs.
     *
     * The bitmap is *not* recycled here — the caller owns it.
     */
    fun sendFrame(
        bitmap: Bitmap?,
        xml: String,
        topPackage: String,
        activityName: String,
        targetPackage: String,
        isFirstScreen: Boolean = false
    ): FrameSendResult {
        if (!connected) return FrameSendResult(false, false)

        // JPEG encoding is slow and touches no wire state: keep it out of the lock.
        var imageBytes: ByteArray? = null
        if (bitmap != null) {
            try {
                val baos = ByteArrayOutputStream()
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, baos)
                imageBytes = baos.toByteArray()
            } catch (e: IOException) {
                Log.e(TAG, "sendFrame screenshot compression failed: ${e.message}")
                connected = false
                return FrameSendResult(false, false)
            } catch (e: Exception) {
                // Degrade to XML-only: the connection is still usable.
                Log.e(TAG, "sendFrame screenshot error: ${e.message}")
            }
        }
        val img = imageBytes

        var screenshotSent = false
        var xmlSent = false
        try {
            val xmlBytes = xml.toByteArray(StandardCharsets.UTF_8)
            // Hashing is pure CPU work: do it before taking the lock, but keep the
            // *store* inside the lock so "recorded hash == last xml on the wire"
            // can never be reordered by two overlapping sends.
            val xmlHash = md5Hex(xml)

            synchronized(writeLock) {
                val out = dos ?: return FrameSendResult(false, false)

                if (img != null) {
                    out.writeByte('S'.code)
                    out.write("${img.size}\n".toByteArray(StandardCharsets.UTF_8))
                    out.write(img)
                    out.flush()
                    screenshotSent = true
                }

                out.writeByte('X'.code)
                out.write("$topPackage\n".toByteArray(StandardCharsets.UTF_8))
                out.write("$activityName\n".toByteArray(StandardCharsets.UTF_8))
                out.write("$targetPackage\n".toByteArray(StandardCharsets.UTF_8))
                out.write("${if (isFirstScreen) "1" else "0"}\n".toByteArray(StandardCharsets.UTF_8))
                out.write("${xmlBytes.size}\n".toByteArray(StandardCharsets.UTF_8))
                out.write(xmlBytes)
                out.flush()
                xmlSent = true

                // Still holding writeLock: the frame is on the wire, so record it
                // now. A later sendFrame cannot overtake this store.
                lastSentXmlHash = xmlHash
                lastFrameSentAt = System.currentTimeMillis()
            }
            Log.d(
                TAG,
                "Frame sent: screenshot=${img?.size ?: 0} bytes, xml=${xmlBytes.size} bytes " +
                    "(top=$topPackage, activity=$activityName)"
            )
        } catch (e: IOException) {
            Log.e(TAG, "sendFrame failed: ${e.message}")
            connected = false
        } catch (e: Exception) {
            Log.e(TAG, "sendFrame error: ${e.message}")
        }
        return FrameSendResult(screenshotSent, xmlSent)
    }

    private fun md5Hex(s: String): String {
        val bytes = MessageDigest.getInstance("MD5").digest(s.toByteArray(StandardCharsets.UTF_8))
        return bytes.joinToString("") { "%02x".format(it) }
    }

    fun sendExternalApp(topPackage: String, targetPackage: String): Boolean {
        if (!connected) return false
        return try {
            val json = """{"detected_package":"$topPackage","target_package":"$targetPackage"}"""

            synchronized(writeLock) {
                val out = dos ?: return false
                out.writeByte('E'.code)
                out.write("$json\n".toByteArray(StandardCharsets.UTF_8))
                out.flush()
            }
            Log.d(TAG, "ExternalApp sent: $topPackage")
            true
        } catch (e: IOException) {
            Log.e(TAG, "sendExternalApp failed: ${e.message}")
            connected = false
            false
        } catch (e: Exception) {
            Log.e(TAG, "sendExternalApp error: ${e.message}")
            false
        }
    }

    fun sendPackageName(targetPackage: String): Boolean {
        if (!connected) return false
        return try {
            synchronized(writeLock) {
                val out = dos ?: return false
                out.writeByte('P'.code)
                out.write("$targetPackage\n".toByteArray(StandardCharsets.UTF_8))
                out.flush()
            }
            Log.d(TAG, "PackageName sent: $targetPackage")
            true
        } catch (e: IOException) {
            Log.e(TAG, "sendPackageName failed: ${e.message}")
            connected = false
            false
        } catch (e: Exception) {
            Log.e(TAG, "sendPackageName error: ${e.message}")
            false
        }
    }

    fun sendFinish(): Boolean {
        if (!connected) return false
        return try {
            synchronized(writeLock) {
                val out = dos ?: return false
                out.writeByte('F'.code)
                out.flush()
            }
            Log.d(TAG, "Finish signal sent")
            true
        } catch (e: IOException) {
            Log.e(TAG, "sendFinish failed: ${e.message}")
            connected = false
            false
        } catch (e: Exception) {
            Log.e(TAG, "sendFinish error: ${e.message}")
            false
        }
    }

    fun sendNoChange(): Boolean {
        if (!connected) return false
        return try {
            synchronized(writeLock) {
                val out = dos ?: return false
                out.writeByte('N'.code)
                out.flush()
            }
            Log.d(TAG, "NoChange signal sent")
            true
        } catch (e: IOException) {
            Log.e(TAG, "sendNoChange failed: ${e.message}")
            connected = false
            false
        } catch (e: Exception) {
            Log.e(TAG, "sendNoChange error: ${e.message}")
            false
        }
    }
}
