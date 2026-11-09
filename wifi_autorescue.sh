#!/bin/bash
# Continuous WiFi Watchdog for GreenBots Hotspot

TARGET_SSID="GreenBots"
LOG_TAG="wifi-autorescue"

logger -t "$LOG_TAG" "Starting GreenBots auto-rescue watchdog daemon..."

# Ensure power saving is permanently disabled
iw dev wlan0 set power_save off 2>/dev/null

while true; do
    WLAN_IP=$(ip -4 addr show wlan0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
    
    if [ -z "$WLAN_IP" ]; then
        logger -t "$LOG_TAG" "wlan0 has no IP address. Rescanning and reconnecting to $TARGET_SSID..."
        iw dev wlan0 set power_save off 2>/dev/null
        nmcli dev wifi rescan 2>/dev/null
        sleep 2
        nmcli con up "$TARGET_SSID" 2>/dev/null
    else
        PS_STATE=$(iw dev wlan0 get power_save 2>/dev/null)
        if echo "$PS_STATE" | grep -qi "on"; then
            logger -t "$LOG_TAG" "Power save was ON. Forcing OFF..."
            iw dev wlan0 set power_save off 2>/dev/null
        fi
    fi
    sleep 8
done
