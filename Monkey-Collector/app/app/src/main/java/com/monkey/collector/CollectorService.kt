package com.monkey.collector

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.AccessibilityServiceInfo
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import android.view.accessibility.AccessibilityWindowInfo
import androidx.core.app.NotificationCompat

class CollectorService : AccessibilityService() {

    companion object {
        private const val TAG = "CollectorService"
        private const val DEBOUNCE_MS = 300L
        private const val MIN_CAPTURE_INTERVAL_MS = 3000L
        private const val NOTIFICATION_CHANNEL_ID = "MonkeyCollector_Channel"
        private const val NOTIFICATION_ID = 1

        private val EXCLUDED_PACKAGES = setOf(
            "com.android.systemui",
            "com.android.permissioncontroller",
            "com.monkey.collector",
            // Google Play Services / sign-in & install surfaces: these are
            // unavoidable hand-offs for sign-in-gated apps (e.g. Google Docs on
            // a signed-out device). Treating them as "external apps to flee"
            // produced an endless E -> BACK -> relaunch -> gms storm, so skip
            // them instead (the session ends cleanly on timeout).
            "com.google.android.gms",
            "com.google.android.gsf",
            "com.android.vending"
        )

        private val LAUNCHER_PACKAGES = setOf(
            "com.google.android.apps.nexuslauncher",
            "com.android.launcher3",
            "com.android.launcher",
            "com.sec.android.app.launcher",
            "com.huawei.android.launcher",
            "com.miui.home",
            "com.oppo.launcher",
            "com.vivo.launcher",
        )

        private fun isLauncher(pkg: String): Boolean {
            return pkg in LAUNCHER_PACKAGES || pkg.contains("launcher", ignoreCase = true)
        }

        var instance: CollectorService? = null
            private set
    }

    private var tcpClient: TcpClient? = null
    private var targetPackage: String = ""
    private var currentActivityName: String = ""
    private var stepCount: Int = 0
    private var lastEventTime: Long = 0
    private var consecutiveBackCount: Int = 0
    private var isCollecting: Boolean = false
    @Volatile private var lastCaptureTime: Long = 0
    private var screenStabilizer: ScreenStabilizer? = null

    // Standby loop state: keeps a TCP connection to the server so the server
    // can push START messages whenever it wants the client to begin collecting.
    private var standbyThread: Thread? = null
    @Volatile private var shutdownRequested: Boolean = false

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this

        serviceInfo = serviceInfo.apply {
            eventTypes = AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED or
                    AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED
            feedbackType = AccessibilityServiceInfo.FEEDBACK_GENERIC
            flags = AccessibilityServiceInfo.FLAG_REPORT_VIEW_IDS or
                    AccessibilityServiceInfo.FLAG_RETRIEVE_INTERACTIVE_WINDOWS or
                    AccessibilityServiceInfo.FLAG_INCLUDE_NOT_IMPORTANT_VIEWS
            notificationTimeout = DEBOUNCE_MS
        }

        // If the user has already configured the server in a previous session,
        // start connecting immediately so the server can drive the device.
        val prefs = getSharedPreferences("collector_settings", Context.MODE_PRIVATE)
        val savedIp = prefs.getString("server_ip", "") ?: ""
        if (savedIp.isNotEmpty()) {
            beginStandby()
        }

        Log.i(TAG, "Service connected")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        if (event == null) return

        // Track Activity name from WINDOW_STATE_CHANGED (even before isCollecting check)
        if (event.eventType == AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED) {
            val pkg = event.packageName?.toString()
            val cls = event.className?.toString()
            if (pkg != null && cls != null && pkg !in EXCLUDED_PACKAGES
                && cls.contains(".") && !cls.startsWith("android.widget.")
            ) {
                currentActivityName = "$pkg/$cls"
                Log.d(TAG, "Activity changed: $currentActivityName")
            }
        }

        if (!isCollecting) return

        val eventType = event.eventType
        if (eventType != AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED &&
            eventType != AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED
        ) return

        // Debounce
        val now = System.currentTimeMillis()
        if (now - lastEventTime < DEBOUNCE_MS) return
        lastEventTime = now

        // Check TCP connection
        if (tcpClient?.isConnected() != true) {
            Log.w(TAG, "TCP not connected, skipping capture")
            return
        }

        // Get top interactable window root (MobileGPT-V2 pattern)
        val topResult = getTopInteractableRoot()
        if (topResult == null) {
            Log.d(TAG, "No interactable window found")
            return
        }
        val (topPackage, root) = topResult

        // Check for external app
        if (targetPackage.isNotEmpty() &&
            topPackage != targetPackage &&
            topPackage !in EXCLUDED_PACKAGES
        ) {
            Thread {
                val sent = tcpClient?.sendExternalApp(topPackage, targetPackage) ?: false
                if (!sent) {
                    Log.w(TAG, "Failed to send external app event")
                }
            }.start()

            consecutiveBackCount++

            if (consecutiveBackCount >= 3 || isLauncher(topPackage)) {
                try {
                    val launchIntent = packageManager.getLaunchIntentForPackage(targetPackage)
                    if (launchIntent != null) {
                        launchIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                        startActivity(launchIntent)
                    } else {
                        Runtime.getRuntime().exec(
                            arrayOf("am", "start",
                                "-a", "android.intent.action.MAIN",
                                "-c", "android.intent.category.LAUNCHER",
                                targetPackage)
                        )
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Force launch failed: ${e.message}")
                }
                consecutiveBackCount = 0
            } else {
                performGlobalAction(GLOBAL_ACTION_BACK)
            }
            try { root.recycle() } catch (_: Exception) {}
            return
        }

        consecutiveBackCount = 0

        Thread {
            try {
                // Step 1: Wait for screen to stabilize (visual bitmap comparison)
                val stabilizer = screenStabilizer
                if (stabilizer != null) {
                    val stabilized = stabilizer.waitForStable()
                    if (!stabilized) {
                        Log.w(TAG, "Screen stabilization timeout, capturing anyway")
                        // Stabilization 실패 시 최소 캡처 간격 보장 (폭주 방지)
                        val elapsed = System.currentTimeMillis() - lastCaptureTime
                        if (elapsed < MIN_CAPTURE_INTERVAL_MS) {
                            return@Thread
                        }
                    }

                    // Step 2: Check for actual visual change
                    if (!stabilizer.hasVisualChange()) {
                        Log.d(TAG, "No visual change detected, sending N signal")
                        tcpClient?.sendNoChange()
                        return@Thread
                    }
                }

                // Step 2.5: First screen detection
                val isFirstScreen = if (stabilizer != null) {
                    if (stepCount == 0) stabilizer.saveFirstScreen()
                    stabilizer.isFirstScreen()
                } else {
                    false
                }

                // Step 3: Take screenshot
                val bitmap = ScreenCapture.takeSync(this)

                // Step 4: Dump XML (existing logic)
                val xml = XmlDumper.dumpNodeTree(root)

                // Step 5: Send data with return value check
                if (bitmap != null) {
                    val screenshotSent = tcpClient?.sendScreenshot(bitmap) ?: false
                    if (!screenshotSent) {
                        Log.w(TAG, "Failed to send screenshot at step $stepCount")
                    }
                    bitmap.recycle()
                }

                val activityAtCapture = currentActivityName
                val xmlSent = tcpClient?.sendXml(xml, topPackage, activityAtCapture, targetPackage, isFirstScreen) ?: false
                if (!xmlSent) {
                    Log.w(TAG, "Failed to send XML at step $stepCount")
                }

                stepCount++
                lastCaptureTime = System.currentTimeMillis()
                Log.d(TAG, "Step $stepCount captured for $topPackage")

            } catch (e: Exception) {
                Log.e(TAG, "Capture error: ${e.message}")
            } finally {
                try { root.recycle() } catch (_: Exception) {}
            }
        }.start()
    }

    override fun onInterrupt() {
        Log.w(TAG, "Service interrupted")
    }

    override fun onDestroy() {
        super.onDestroy()
        instance = null
        shutdownRequested = true
        stopStandby()
        stopCollection()
        screenStabilizer?.release()
        screenStabilizer = null
    }

    fun startCollection(
        serverIp: String,
        serverPort: Int,
        targetPkg: String,
        screenWidth: Int,
        screenHeight: Int,
        screenDensityDpi: Int,
        existingClient: TcpClient? = null,
    ) {
        targetPackage = targetPkg
        currentActivityName = ""
        stepCount = 0
        consecutiveBackCount = 0

        // Start foreground service (required before MediaProjection on API 29+)
        startForegroundService()

        // Initialize screen stabilizer with MediaProjection (reuse if exists)
        if (MediaProjectionHelper.isGranted) {
            if (screenStabilizer == null) {
                val stabilizer = ScreenStabilizer(screenWidth, screenHeight, screenDensityDpi)
                stabilizer.initProjection(this)
                screenStabilizer = stabilizer
            }
            screenStabilizer!!.startCaptureSession(
                MediaProjectionHelper.resultCode,
                MediaProjectionHelper.resultData!!
            )
            // VirtualDisplay가 첫 프레임을 렌더링할 시간 확보
            Thread.sleep(500)
            Log.i(TAG, "ScreenStabilizer initialized (${screenWidth}x${screenHeight})")
        } else {
            Log.w(TAG, "MediaProjection not granted, running without visual stabilization")
        }

        // Reuse the standby connection if the server already pushed us a START;
        // otherwise open a fresh socket (kept for API symmetry).
        val client = existingClient ?: TcpClient(serverIp, serverPort)
        tcpClient = client
        client.setOnSessionEnd {
            Log.i(TAG, "Server ended session, stopping collection")
            Handler(Looper.getMainLooper()).post { stopCollection() }
        }
        Thread {
            val connected = client.isConnected() || client.connect()
            if (connected) {
                client.sendPackageName(targetPackage)
                isCollecting = true
                Log.i(TAG, "Collection started: target=$targetPkg, server=$serverIp:$serverPort")
            } else {
                Log.e(TAG, "TCP connection failed, collection NOT started")
            }
        }.start()
    }

    fun stopCollection() {
        isCollecting = false

        // Pause screen stabilizer (keep MediaProjection alive for reuse)
        screenStabilizer?.stopCaptureSession()

        // Null out tcpClient immediately to prevent in-flight worker threads
        // from sending more data (they use tcpClient?.send* null-safe calls)
        val client = tcpClient
        tcpClient = null
        Thread {
            client?.sendFinish()
            Thread.sleep(200)
            client?.disconnect()
        }.start()

        // Stop foreground
        stopForeground(STOP_FOREGROUND_REMOVE)

        Log.i(TAG, "Collection stopped. Steps: $stepCount")
    }

    /**
     * Start (or restart) the server standby loop.
     *
     * Holds a persistent TCP connection to the server so it can push START
     * messages — those trigger [startCollection] with the server-chosen
     * target package.  Reconnects automatically if the socket drops.
     */
    fun beginStandby() {
        if (shutdownRequested) return
        if (standbyThread?.isAlive == true) return

        val prefs = getSharedPreferences("collector_settings", Context.MODE_PRIVATE)
        val ip = prefs.getString("server_ip", "") ?: ""
        val port = prefs.getInt("server_port", 12345)
        if (ip.isEmpty()) {
            Log.w(TAG, "Server IP not configured; standby skipped")
            return
        }

        standbyThread = Thread {
            while (!shutdownRequested) {
                if (isCollecting) {
                    // A session is in progress; wait for stopCollection() to
                    // release the connection before attempting a new standby.
                    try { Thread.sleep(1000) } catch (_: InterruptedException) { break }
                    continue
                }

                val client = TcpClient(ip, port)
                client.setOnStart { pkg ->
                    Log.i(TAG, "Standby: server START for $pkg")
                    val metrics = resources.displayMetrics
                    Handler(Looper.getMainLooper()).post {
                        startCollection(
                            serverIp = ip,
                            serverPort = port,
                            targetPkg = pkg,
                            screenWidth = metrics.widthPixels,
                            screenHeight = metrics.heightPixels,
                            screenDensityDpi = metrics.densityDpi,
                            existingClient = client,
                        )
                    }
                }
                client.setOnSessionEnd {
                    Handler(Looper.getMainLooper()).post { stopCollection() }
                }

                if (client.connect()) {
                    Log.i(TAG, "Standby: connected to $ip:$port, waiting for START")
                    // Block until the connection drops (reader thread handles messages).
                    while (client.isConnected() && !shutdownRequested && !isCollecting) {
                        try { Thread.sleep(500) } catch (_: InterruptedException) { break }
                    }
                } else {
                    Log.w(TAG, "Standby: connect failed, retrying in 3s")
                }

                if (!shutdownRequested && !isCollecting) {
                    try { Thread.sleep(3000) } catch (_: InterruptedException) { break }
                }
            }
            Log.i(TAG, "Standby thread exiting")
        }.apply {
            isDaemon = true
            name = "CollectorService-Standby"
            start()
        }
    }

    fun stopStandby() {
        standbyThread?.interrupt()
        standbyThread = null
    }

    private fun startForegroundService() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                NOTIFICATION_CHANNEL_ID,
                "Monkey-Collector Service",
                NotificationManager.IMPORTANCE_LOW
            )
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }

        val notification = NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setContentTitle("Monkey-Collector")
            .setContentText("Collecting UI data...")
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()

        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                startForeground(
                    NOTIFICATION_ID,
                    notification,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
                )
            } else {
                startForeground(NOTIFICATION_ID, notification)
            }
            Log.d(TAG, "Foreground service started")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start foreground service: ${e.message}")
        }
    }

    /**
     * Get the top interactable application window's root node and package name.
     * Iterates through windows to find TYPE_APPLICATION windows, excluding system packages.
     * (MobileGPT-V2 getTopInteractableRoot pattern)
     */
    private fun getTopInteractableRoot(): Pair<String, AccessibilityNodeInfo>? {
        return try {
            val windowList = windows ?: return null

            if (Log.isLoggable(TAG, Log.DEBUG)) {
                for (w in windowList) {
                    Log.d(TAG, "Window: type=${w.type}, layer=${w.layer}, " +
                            "pkg=${w.root?.packageName}, active=${w.isActive}, " +
                            "focused=${w.isFocused}")
                }
            }

            for (w in windowList) {
                if (w.type != AccessibilityWindowInfo.TYPE_APPLICATION) continue
                val root = w.root ?: continue
                val pkg = root.packageName?.toString() ?: continue
                if (pkg in EXCLUDED_PACKAGES) {
                    root.recycle()
                    continue
                }
                return Pair(pkg, root)
            }
            null
        } catch (e: Exception) {
            Log.e(TAG, "getTopInteractableRoot error: ${e.message}")
            null
        }
    }

    /**
     * Get the package name of the current foreground app.
     */
    fun getCurrentForegroundPackage(): String? {
        return try {
            val windowList = windows ?: return null
            for (w in windowList) {
                if (w.type != AccessibilityWindowInfo.TYPE_APPLICATION) continue
                val root = w.root ?: continue
                val pkg = root.packageName?.toString() ?: continue
                root.recycle()
                if (pkg in EXCLUDED_PACKAGES || isLauncher(pkg)) continue
                return pkg
            }
            null
        } catch (e: Exception) {
            null
        }
    }

}
