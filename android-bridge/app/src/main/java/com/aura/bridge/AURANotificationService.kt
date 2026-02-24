// AURA Notification Service - Captures all notifications
// File: app/src/main/java/com/aura/bridge/AURANotificationService.kt

package com.aura.bridge

import android.app.Notification
import android.content.Intent
import android.os.Bundle
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import android.util.Log
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import org.java_websocket.server.WebSocketServer
import org.java_websocket.WebSocket
import org.java_websocket.handshake.ClientHandshake
import org.json.JSONObject
import java.net.InetSocketAddress
import java.util.concurrent.CopyOnWriteArrayList

class AURANotificationService : NotificationListenerService() {
    
    companion object {
        const val TAG = "AURANotifications"
        var instance: AURANotificationService? = null
        val activeNotifications = mutableListOf<NotificationData>()
        val webSocketClients = CopyOnWriteArrayList<WebSocket>()
    }
    
    private var webSocketServer: NotificationWebSocketServer? = null
    
    data class NotificationData(
        val key: String,
        val packageName: String,
        val title: String,
        val text: String,
        val bigText: String?,
        val timestamp: Long,
        val canReply: Boolean,
        val actions: List<NotificationAction>
    )
    
    data class NotificationAction(
        val title: String,
        val remoteInputs: List<String>? = null
    )
    
    override fun onCreate() {
        super.onCreate()
        instance = this
        startWebSocketServer()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        instance = null
        webSocketServer?.stop()
    }
    
    override fun onListenerConnected() {
        super.onListenerConnected()
        Log.i(TAG, "Notification listener connected")
        
        // Get existing notifications
        activeNotifications.clear()
        activeNotifications.addAll(activeNotifications.map { parseNotification(it) })
    }
    
    override fun onNotificationPosted(sbn: StatusBarNotification) {
        val notificationData = parseNotification(sbn)
        activeNotifications.add(notificationData)
        
        Log.d(TAG, "Notification posted: ${notificationData.title} - ${notificationData.text}")
        
        // Broadcast to WebSocket clients
        broadcastNotification(notificationData, "posted")
    }
    
    override fun onNotificationRemoved(sbn: StatusBarNotification) {
        val key = sbn.key
        activeNotifications.removeAll { it.key == key }
        
        Log.d(TAG, "Notification removed: $key")
        
        // Broadcast removal
        broadcastRemoval(key)
    }
    
    private fun parseNotification(sbn: StatusBarNotification): NotificationData {
        val extras = sbn.notification.extras
        
        // Extract title
        val title = extras.getString(Notification.EXTRA_TITLE) 
            ?: extras.getString("android.title") 
            ?: ""
        
        // Extract text
        val text = extras.getCharSequence(Notification.EXTRA_TEXT)?.toString() ?: ""
        val bigText = extras.getCharSequence(Notification.EXTRA_BIG_TEXT)?.toString()
        
        // Check if can reply
        val wearableExtender = Notification.WearableExtender(sbn.notification)
        val canReply = wearableExtender.actions.any { it.remoteInputs != null }
        
        // Parse actions
        val actions = sbn.notification.actions?.map { action ->
            NotificationAction(
                title = action.title?.toString() ?: "",
                remoteInputs = action.remoteInputs?.map { it.resultKey }
            )
        } ?: emptyList()
        
        return NotificationData(
            key = sbn.key,
            packageName = sbn.packageName,
            title = title,
            text = text,
            bigText = bigText,
            timestamp = sbn.postTime,
            canReply = canReply || actions.any { it.remoteInputs != null },
            actions = actions
        )
    }
    
    private fun broadcastNotification(data: NotificationData, eventType: String) {
        val json = JSONObject().apply {
            put("type", "notification")
            put("event", eventType)
            put("key", data.key)
            put("package", data.packageName)
            put("title", data.title)
            put("text", data.text)
            put("big_text", data.bigText ?: "")
            put("timestamp", data.timestamp)
            put("can_reply", data.canReply)
        }
        
        val message = json.toString()
        
        GlobalScope.launch {
            webSocketClients.forEach { client ->
                try {
                    client.send(message)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to send to WebSocket client", e)
                }
            }
        }
    }
    
    private fun broadcastRemoval(key: String) {
        val json = JSONObject().apply {
            put("type", "notification")
            put("event", "removed")
            put("key", key)
        }
        
        val message = json.toString()
        
        GlobalScope.launch {
            webSocketClients.forEach { client ->
                try {
                    client.send(message)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to send removal to WebSocket client", e)
                }
            }
        }
    }
    
    private fun startWebSocketServer() {
        try {
            webSocketServer = NotificationWebSocketServer(8081)
            webSocketServer?.start()
            Log.i(TAG, "WebSocket server started on port 8081")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start WebSocket server", e)
        }
    }
    
    // Reply to notification programmatically
    fun replyToNotification(key: String, message: String): Boolean {
        val sbn = activeNotifications.find { it.key == key }
            ?: return false
        
        // Get actual StatusBarNotification
        val statusBarNotification = getActiveNotifications().find { it.key == key }
            ?: return false
        
        // Find reply action
        val notification = statusBarNotification.notification
        val wearableExtender = Notification.WearableExtender(notification)
        
        wearableExtender.actions.forEach { action ->
            if (action.remoteInputs != null && action.remoteInputs.isNotEmpty()) {
                val intent = Intent()
                val bundle = Bundle()
                
                action.remoteInputs.forEach { remoteInput ->
                    bundle.putCharSequence(remoteInput.resultKey, message)
                }
                
                RemoteInput.addResultsToIntent(action.remoteInputs, intent, bundle)
                
                try {
                    action.actionIntent.send(this, 0, intent)
                    return true
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to send reply", e)
                }
            }
        }
        
        // Try regular actions
        notification.actions?.forEach { action ->
            if (action.remoteInputs != null && action.remoteInputs.isNotEmpty()) {
                val intent = Intent()
                val bundle = Bundle()
                
                action.remoteInputs.forEach { remoteInput ->
                    bundle.putCharSequence(remoteInput.resultKey, message)
                }
                
                RemoteInput.addResultsToIntent(action.remoteInputs, intent, bundle)
                
                try {
                    action.actionIntent.send(this, 0, intent)
                    return true
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to send reply via regular action", e)
                }
            }
        }
        
        return false
    }
    
    // Get all active notifications
    fun getNotifications(): List<NotificationData> {
        return activeNotifications.toList()
    }
    
    // Inner WebSocket server class
    inner class NotificationWebSocketServer(port: Int) : WebSocketServer(InetSocketAddress(port)) {
        
        override fun onOpen(conn: WebSocket, handshake: ClientHandshake) {
            Log.i(TAG, "WebSocket client connected: ${conn.remoteSocketAddress}")
            webSocketClients.add(conn)
            
            // Send current notifications to new client
            activeNotifications.forEach { notification ->
                val json = JSONObject().apply {
                    put("type", "notification")
                    put("event", "posted")
                    put("key", notification.key)
                    put("package", notification.packageName)
                    put("title", notification.title)
                    put("text", notification.text)
                    put("big_text", notification.bigText ?: "")
                    put("timestamp", notification.timestamp)
                    put("can_reply", notification.canReply)
                }
                conn.send(json.toString())
            }
        }
        
        override fun onClose(conn: WebSocket, code: Int, reason: String, remote: Boolean) {
            Log.i(TAG, "WebSocket client disconnected: ${conn.remoteSocketAddress}")
            webSocketClients.remove(conn)
        }
        
        override fun onMessage(conn: WebSocket, message: String) {
            // Handle commands from clients
            try {
                val json = JSONObject(message)
                val action = json.optString("action")
                
                when (action) {
                    "reply" -> {
                        val key = json.getString("key")
                        val replyMessage = json.getString("message")
                        val success = replyToNotification(key, replyMessage)
                        
                        conn.send(JSONObject().apply {
                            put("type", "reply_result")
                            put("success", success)
                            put("key", key)
                        }.toString())
                    }
                    "clear" -> {
                        val key = json.getString("key")
                        cancelNotification(key)
                        conn.send(JSONObject().apply {
                            put("type", "clear_result")
                            put("success", true)
                            put("key", key)
                        }.toString())
                    }
                    "clear_all" -> {
                        cancelAllNotifications()
                        conn.send(JSONObject().apply {
                            put("type", "clear_all_result")
                            put("success", true)
                        }.toString())
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error handling WebSocket message", e)
            }
        }
        
        override fun onError(conn: WebSocket, ex: Exception) {
            Log.e(TAG, "WebSocket error", ex)
        }
        
        override fun onStart() {
            Log.i(TAG, "WebSocket server started")
        }
    }
}
