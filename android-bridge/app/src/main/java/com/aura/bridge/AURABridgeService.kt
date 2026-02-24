// AURA Android Bridge - Main Service
// File: app/src/main/java/com/aura/bridge/AURABridgeService.kt

package com.aura.bridge

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.app.Notification
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Path
import android.graphics.PixelFormat
import android.graphics.Rect
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import fi.iki.elonen.NanoHTTPD
import kotlinx.coroutines.*
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.net.InetSocketAddress

class AURABridgeService : AccessibilityService() {
    
    companion object {
        const val TAG = "AURABridge"
        const val SERVER_PORT = 8080
    }
    
    private var httpServer: BridgeHttpServer? = null
    private var mediaProjection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null
    private val mainHandler = Handler(Looper.getMainLooper())
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    
    // UI State
    private var lastUIElements: List<UIElement> = emptyList()
    private var screenWidth: Int = 1080
    private var screenHeight: Int = 2400
    
    override fun onServiceConnected() {
        super.onServiceConnected()
        Log.i(TAG, "AURA Bridge Service connected")
        
        // Configure accessibility service
        serviceInfo.apply {
            eventTypes = AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED or
                        AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED or
                        AccessibilityEvent.TYPE_VIEW_CLICKED
            feedbackType = AccessibilityServiceInfo.FEEDBACK_GENERIC
            flags = AccessibilityServiceInfo.FLAG_REPORT_VIEW_IDS or
                   AccessibilityServiceInfo.FLAG_RETRIEVE_INTERACTIVE_WINDOWS
        }
        
        // Get screen dimensions
        val displayMetrics = resources.displayMetrics
        screenWidth = displayMetrics.widthPixels
        screenHeight = displayMetrics.heightPixels
        
        // Start HTTP server
        startHttpServer()
    }
    
    override fun onAccessibilityEvent(event: AccessibilityEvent) {
        when (event.eventType) {
            AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED,
            AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED -> {
                // Update UI tree cache
                updateUIElements()
            }
        }
    }
    
    override fun onInterrupt() {
        Log.w(TAG, "Service interrupted")
    }
    
    override fun onDestroy() {
        super.onDestroy()
        httpServer?.stop()
        scope.cancel()
        mediaProjection?.stop()
        virtualDisplay?.release()
        imageReader?.close()
    }
    
    private fun startHttpServer() {
        try {
            httpServer = BridgeHttpServer(this)
            httpServer?.start(NanoHTTPD.SOCKET_READ_TIMEOUT, false)
            Log.i(TAG, "HTTP server started on port $SERVER_PORT")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start HTTP server", e)
        }
    }
    
    private fun updateUIElements() {
        val rootNode = rootInActiveWindow ?: return
        lastUIElements = parseNodeTree(rootNode)
    }
    
    private fun parseNodeTree(node: AccessibilityNodeInfo, depth: Int = 0): List<UIElement> {
        val elements = mutableListOf<UIElement>()
        
        if (node.isVisibleToUser) {
            val bounds = Rect()
            node.getBoundsInScreen(bounds)
            
            elements.add(UIElement(
                id = node.viewIdResourceName ?: "",
                text = node.text?.toString() ?: "",
                contentDescription = node.contentDescription?.toString() ?: "",
                className = node.className?.toString() ?: "",
                bounds = bounds,
                clickable = node.isClickable,
                editable = node.isEditable,
                scrollable = node.isScrollable,
                selected = node.isSelected
            ))
        }
        
        for (i in 0 until node.childCount) {
            node.getChild(i)?.let {
                elements.addAll(parseNodeTree(it, depth + 1))
            }
        }
        
        return elements
    }
    
    // ==================== COMMAND HANDLERS ====================
    
    fun handleCommand(action: String, params: JSONObject): JSONObject {
        return when (action) {
            "tap" -> handleTap(params)
            "long_press" -> handleLongPress(params)
            "type" -> handleType(params)
            "swipe" -> handleSwipe(params)
            "scroll" -> handleScroll(params)
            "back" -> handleKey("KEYCODE_BACK")
            "home" -> handleKey("KEYCODE_HOME")
            "recent" -> handleKey("KEYCODE_APP_SWITCH")
            "launch_app" -> handleLaunchApp(params)
            "open_url" -> handleOpenUrl(params)
            "send_message" -> handleSendMessage(params)
            "make_call" -> handleMakeCall(params)
            "get_ui_tree" -> handleGetUITree()
            "get_screen_size" -> handleGetScreenSize()
            "screenshot" -> handleScreenshot(params)
            "get_notifications" -> handleGetNotifications()
            "reply_notification" -> handleReplyNotification(params)
            "learn_app" -> handleLearnApp(params)
            "find_element" -> handleFindElement(params)
            else -> JSONObject().put("success", false).put("error", "Unknown action: $action")
        }
    }
    
    private fun handleTap(params: JSONObject): JSONObject {
        val result = JSONObject()
        
        // Try coordinates first
        if (params.has("x") && params.has("y")) {
            val x = params.getInt("x")
            val y = params.getInt("y")
            return performTap(x, y)
        }
        
        // Try to find by text
        val text = params.optString("text", null)
        if (text != null) {
            val element = findElementByText(text)
            if (element != null) {
                return performTap(element.centerX(), element.centerY())
            }
        }
        
        // Try content description
        val contentDesc = params.optString("content_desc", null)
        if (contentDesc != null) {
            val element = findElementByContentDesc(contentDesc)
            if (element != null) {
                return performTap(element.centerX(), element.centerY())
            }
        }
        
        // Try resource ID
        val resourceId = params.optString("resource_id", null)
        if (resourceId != null) {
            val element = findElementByResourceId(resourceId)
            if (element != null) {
                return performTap(element.centerX(), element.centerY())
            }
        }
        
        return result.put("success", false).put("error", "Element not found")
    }
    
    private fun performTap(x: Int, y: Int): JSONObject {
        val result = JSONObject()
        
        val path = Path()
        path.moveTo(x.toFloat(), y.toFloat())
        
        val gesture = GestureDescription.Builder()
            .addStroke(GestureDescription.StrokeDescription(path, 0, 100))
            .build()
        
        val dispatched = dispatchGesture(gesture, object : GestureResultCallback() {
            override fun onCompleted(gestureDescription: GestureDescription?) {
                super.onCompleted(gestureDescription)
            }
        }, null)
        
        return result.put("success", dispatched)
    }
    
    private fun handleLongPress(params: JSONObject): JSONObject {
        val result = JSONObject()
        val duration = params.optInt("duration_ms", 500)
        
        val x = if (params.has("x")) params.getInt("x") else {
            val text = params.optString("text", "")
            val element = findElementByText(text)
            element?.centerX() ?: return result.put("success", false).put("error", "Element not found")
        }
        
        val y = if (params.has("y")) params.getInt("y") else {
            val text = params.optString("text", "")
            val element = findElementByText(text)
            element?.centerY() ?: return result.put("success", false).put("error", "Element not found")
        }
        
        val path = Path()
        path.moveTo(x.toFloat(), y.toFloat())
        
        val gesture = GestureDescription.Builder()
            .addStroke(GestureDescription.StrokeDescription(path, 0, duration.toLong()))
            .build()
        
        val dispatched = dispatchGesture(gesture, null, null)
        return result.put("success", dispatched)
    }
    
    private fun handleType(params: JSONObject): JSONObject {
        val result = JSONObject()
        val text = params.getString("text")
        val resourceId = params.optString("resource_id", null)
        val clearFirst = params.optBoolean("clear_first", true)
        
        val element = if (resourceId != null) {
            findElementByResourceId(resourceId)
        } else {
            // Find focused or first editable field
            lastUIElements.find { it.editable }
        }
        
        if (element == null) {
            return result.put("success", false).put("error", "No editable field found")
        }
        
        // Focus element
        val node = findAccessibilityNode(element)
        node?.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
        
        // Clear existing text if requested
        if (clearFirst) {
            val args = Bundle()
            args.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, "")
            node?.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)
        }
        
        // Type text
        val args = Bundle()
        args.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text)
        val success = node?.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args) ?: false
        
        return result.put("success", success)
    }
    
    private fun handleSwipe(params: JSONObject): JSONObject {
        val result = JSONObject()
        val direction = params.getString("direction")
        val distance = params.optInt("distance", 500)
        
        val startX = params.optInt("start_x", screenWidth / 2)
        val startY = params.optInt("start_y", screenHeight / 2)
        
        val (endX, endY) = when (direction) {
            "up" -> Pair(startX, startY - distance)
            "down" -> Pair(startX, startY + distance)
            "left" -> Pair(startX - distance, startY)
            "right" -> Pair(startX + distance, startY)
            else -> return result.put("success", false).put("error", "Invalid direction")
        }
        
        val path = Path()
        path.moveTo(startX.toFloat(), startY.toFloat())
        path.lineTo(endX.toFloat(), endY.toFloat())
        
        val gesture = GestureDescription.Builder()
            .addStroke(GestureDescription.StrokeDescription(path, 0, 300))
            .build()
        
        val dispatched = dispatchGesture(gesture, null, null)
        return result.put("success", dispatched)
    }
    
    private fun handleScroll(params: JSONObject): JSONObject {
        val result = JSONObject()
        val direction = params.getString("direction")
        
        val rootNode = rootInActiveWindow
        val action = if (direction == "forward" || direction == "down") {
            AccessibilityNodeInfo.ACTION_SCROLL_FORWARD
        } else {
            AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD
        }
        
        // Try to scroll the first scrollable element
        val scrollable = findFirstScrollable(rootNode)
        val success = scrollable?.performAction(action) ?: false
        
        return result.put("success", success)
    }
    
    private fun handleKey(keyCode: String): JSONObject {
        val result = JSONObject()
        
        // Use input command via shell (requires root or specific setup)
        // For non-root, we simulate with accessibility actions
        val success = when (keyCode) {
            "KEYCODE_BACK" -> performGlobalAction(GLOBAL_ACTION_BACK)
            "KEYCODE_HOME" -> performGlobalAction(GLOBAL_ACTION_HOME)
            "KEYCODE_APP_SWITCH" -> performGlobalAction(GLOBAL_ACTION_RECENTS)
            else -> false
        }
        
        return result.put("success", success)
    }
    
    private fun handleLaunchApp(params: JSONObject): JSONObject {
        val result = JSONObject()
        val packageName = params.getString("package")
        
        val intent = packageManager.getLaunchIntentForPackage(packageName)
        if (intent != null) {
            intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK
            startActivity(intent)
            return result.put("success", true)
        }
        
        return result.put("success", false).put("error", "App not found")
    }
    
    private fun handleOpenUrl(params: JSONObject): JSONObject {
        val result = JSONObject()
        val url = params.getString("url")
        val browserPackage = params.optString("browser", null)
        
        val intent = Intent(Intent.ACTION_VIEW, android.net.Uri.parse(url))
        intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK
        
        if (browserPackage != null) {
            intent.setPackage(browserPackage)
        }
        
        startActivity(intent)
        return result.put("success", true)
    }
    
    private fun handleSendMessage(params: JSONObject): JSONObject {
        val result = JSONObject()
        val app = params.getString("app")
        
        when (app) {
            "whatsapp" -> {
                val phone = params.getString("phone")
                val message = params.getString("message")
                val method = params.optString("method", "intent")
                
                if (method == "intent") {
                    // Use WhatsApp's URL scheme
                    val url = "https://api.whatsapp.com/send?phone=$phone&text=${java.net.URLEncoder.encode(message, "UTF-8")}"
                    val intent = Intent(Intent.ACTION_VIEW, android.net.Uri.parse(url))
                    intent.setPackage("com.whatsapp")
                    intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK
                    startActivity(intent)
                    return result.put("success", true).put("method", "intent")
                }
            }
            "sms" -> {
                val number = params.getString("number")
                val message = params.getString("message")
                
                val intent = Intent(Intent.ACTION_SENDTO)
                intent.data = android.net.Uri.parse("smsto:$number")
                intent.putExtra("sms_body", message)
                intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK
                startActivity(intent)
                return result.put("success", true)
            }
        }
        
        return result.put("success", false).put("error", "Unsupported app or method")
    }
    
    private fun handleMakeCall(params: JSONObject): JSONObject {
        val result = JSONObject()
        val number = params.getString("number")
        val autoDial = params.optBoolean("auto_dial", false)
        
        val intent = if (autoDial) {
            Intent(Intent.ACTION_CALL, android.net.Uri.parse("tel:$number"))
        } else {
            Intent(Intent.ACTION_DIAL, android.net.Uri.parse("tel:$number"))
        }
        
        intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK
        startActivity(intent)
        return result.put("success", true)
    }
    
    private fun handleGetUITree(): JSONObject {
        val result = JSONObject()
        val elementsArray = JSONArray()
        
        lastUIElements.forEach { element ->
            elementsArray.put(JSONObject().apply {
                put("id", element.id)
                put("text", element.text)
                put("content_description", element.contentDescription)
                put("class_name", element.className)
                put("bounds", JSONObject().apply {
                    put("left", element.bounds.left)
                    put("top", element.bounds.top)
                    put("right", element.bounds.right)
                    put("bottom", element.bounds.bottom)
                })
                put("clickable", element.clickable)
                put("editable", element.editable)
                put("scrollable", element.scrollable)
                put("selected", element.selected)
            })
        }
        
        return result
            .put("success", true)
            .put("elements", elementsArray)
            .put("element_count", elementsArray.length())
    }
    
    private fun handleGetScreenSize(): JSONObject {
        return JSONObject()
            .put("success", true)
            .put("size", JSONObject().apply {
                put("width", screenWidth)
                put("height", screenHeight)
            })
    }
    
    private fun handleScreenshot(params: JSONObject): JSONObject {
        val result = JSONObject()
        val useOcr = params.optBoolean("ocr", false)
        
        // This requires MediaProjection setup
        // For now, return error - implement with proper media projection flow
        return result.put("success", false).put("error", "Screenshot requires MediaProjection setup")
    }
    
    private fun handleGetNotifications(): JSONObject {
        val result = JSONObject()
        // This requires NotificationListenerService
        // Return empty list if not available
        return result.put("success", true).put("notifications", JSONArray())
    }
    
    private fun handleReplyNotification(params: JSONObject): JSONObject {
        val result = JSONObject()
        val key = params.getString("key")
        val message = params.getString("message")
        
        // Requires NotificationListenerService
        return result.put("success", false).put("error", "Notification reply requires NotificationListenerService")
    }
    
    private fun handleLearnApp(params: JSONObject): JSONObject {
        val result = JSONObject()
        val packageName = params.getString("package")
        
        // Launch app
        val intent = packageManager.getLaunchIntentForPackage(packageName)
        if (intent != null) {
            intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK
            startActivity(intent)
            
            // Wait for UI to load
            Thread.sleep(2000)
            updateUIElements()
            
            // Analyze and categorize elements
            val elements = JSONObject()
            lastUIElements.forEachIndexed { index, element ->
                val type = classifyElement(element)
                val key = "${type}_$index"
                elements.put(key, JSONObject().apply {
                    put("type", type)
                    put("resource_id", element.id)
                    put("text_pattern", element.text)
                    put("content_desc_pattern", element.contentDescription)
                    put("class_name", element.className)
                    put("relative_x", element.centerX().toFloat() / screenWidth)
                    put("relative_y", element.centerY().toFloat() / screenHeight)
                })
            }
            
            // Get current activity
            val activityName = getCurrentActivityName()
            
            return result
                .put("success", true)
                .put("package", packageName)
                .put("activity", activityName)
                .put("elements", elements)
        }
        
        return result.put("success", false).put("error", "Failed to launch app")
    }
    
    private fun handleFindElement(params: JSONObject): JSONObject {
        val result = JSONObject()
        val text = params.optString("text", null)
        val resourceId = params.optString("resource_id", null)
        
        val element = when {
            text != null -> findElementByText(text)
            resourceId != null -> findElementByResourceId(resourceId)
            else -> null
        }
        
        if (element != null) {
            return result.put("success", true).put("element", JSONObject().apply {
                put("id", element.id)
                put("text", element.text)
                put("content_description", element.contentDescription)
                put("class_name", element.className)
                put("bounds", JSONObject().apply {
                    put("left", element.bounds.left)
                    put("top", element.bounds.top)
                    put("right", element.bounds.right)
                    put("bottom", element.bounds.bottom)
                })
                put("center_x", element.centerX())
                put("center_y", element.centerY())
            })
        }
        
        return result.put("success", false).put("error", "Element not found")
    }
    
    // ==================== HELPER METHODS ====================
    
    private fun findElementByText(text: String): UIElement? {
        return lastUIElements.find {
            it.text.contains(text, ignoreCase = true)
        }
    }
    
    private fun findElementByContentDesc(desc: String): UIElement? {
        return lastUIElements.find {
            it.contentDescription.contains(desc, ignoreCase = true)
        }
    }
    
    private fun findElementByResourceId(resourceId: String): UIElement? {
        return lastUIElements.find {
            it.id == resourceId
        }
    }
    
    private fun findFirstScrollable(node: AccessibilityNodeInfo?): AccessibilityNodeInfo? {
        if (node == null) return null
        if (node.isScrollable) return node
        
        for (i in 0 until node.childCount) {
            node.getChild(i)?.let {
                val found = findFirstScrollable(it)
                if (found != null) return found
            }
        }
        
        return null
    }
    
    private fun findAccessibilityNode(element: UIElement): AccessibilityNodeInfo? {
        val rootNode = rootInActiveWindow ?: return null
        return findNodeByBounds(rootNode, element.bounds)
    }
    
    private fun findNodeByBounds(node: AccessibilityNodeInfo, bounds: Rect): AccessibilityNodeInfo? {
        val nodeBounds = Rect()
        node.getBoundsInScreen(nodeBounds)
        
        if (nodeBounds == bounds) return node
        
        for (i in 0 until node.childCount) {
            node.getChild(i)?.let {
                val found = findNodeByBounds(it, bounds)
                if (found != null) return found
            }
        }
        
        return null
    }
    
    private fun classifyElement(element: UIElement): String {
        return when {
            element.className.contains("Button", ignoreCase = true) -> "button"
            element.className.contains("EditText", ignoreCase = true) -> "input"
            element.className.contains("TextView", ignoreCase = true) -> {
                if (element.editable) "input" else "label"
            }
            element.className.contains("Image", ignoreCase = true) -> "image"
            element.className.contains("RecyclerView", ignoreCase = true) -> "list"
            element.className.contains("ScrollView", ignoreCase = true) -> "scrollable"
            element.clickable -> "clickable"
            else -> "unknown"
        }
    }
    
    private fun getCurrentActivityName(): String {
        val rootNode = rootInActiveWindow
        return rootNode?.packageName?.toString() ?: ""
    }
    
    // ==================== DATA CLASSES ====================
    
    data class UIElement(
        val id: String,
        val text: String,
        val contentDescription: String,
        val className: String,
        val bounds: Rect,
        val clickable: Boolean,
        val editable: Boolean,
        val scrollable: Boolean = false,
        val selected: Boolean = false
    ) {
        fun centerX(): Int = (bounds.left + bounds.right) / 2
        fun centerY(): Int = (bounds.top + bounds.bottom) / 2
    }
}

// ==================== HTTP SERVER ====================

class BridgeHttpServer(private val service: AURABridgeService) : NanoHTTPD(AURABridgeService.SERVER_PORT) {
    
    override fun serve(session: IHTTPSession): Response {
        val uri = session.uri
        val method = session.method
        
        return when {
            uri == "/health" -> newFixedLengthResponse(
                Response.Status.OK,
                "application/json",
                JSONObject().put("status", "ok").toString()
            )
            
            uri == "/command" && method == Method.POST -> handleCommand(session)
            
            uri == "/screenshot" -> handleScreenshot()
            
            uri == "/ui_tree" -> {
                val result = service.handleCommand("get_ui_tree", JSONObject())
                newFixedLengthResponse(
                    Response.Status.OK,
                    "application/json",
                    result.toString()
                )
            }
            
            else -> newFixedLengthResponse(
                Response.Status.NOT_FOUND,
                "application/json",
                JSONObject().put("error", "Not found").toString()
            )
        }
    }
    
    private fun handleCommand(session: IHTTPSession): Response {
        return try {
            val body = mutableMapOf<String, String>()
            session.parseBody(body)
            
            val jsonBody = JSONObject(body["postData"] ?: "{}")
            val action = jsonBody.getString("action")
            val params = jsonBody.optJSONObject("params") ?: JSONObject()
            
            val result = service.handleCommand(action, params)
            
            newFixedLengthResponse(
                Response.Status.OK,
                "application/json",
                result.toString()
            )
        } catch (e: Exception) {
            newFixedLengthResponse(
                Response.Status.BAD_REQUEST,
                "application/json",
                JSONObject().put("success", false).put("error", e.message).toString()
            )
        }
    }
    
    private fun handleScreenshot(): Response {
        // Return error for now - implement with proper media projection
        return newFixedLengthResponse(
            Response.Status.SERVICE_UNAVAILABLE,
            "application/json",
            JSONObject().put("error", "Screenshot not implemented").toString()
        )
    }
}
