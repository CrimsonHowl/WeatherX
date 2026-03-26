import requests
import json
import os
import time

class WeatherXCloud:
    """
    Handles data synchronization between the Local RPi 3B and ThingSpeak Cloud.
    """
    def __init__(self, api_key="YOUR_THINGSPEAK_WRITE_KEY"):
        self.api_key = api_key
        self.base_url = "https://api.thingspeak.com/update"

    def upload_seed_metrics(self, temp, hum, pres, rain, gtemp=None):
        """
        Uploads core BME280 metrics to ThingSpeak fields.
        Field 1: Temperature
        Field 2: Humidity
        Field 3: Pressure
        Field 4: Precipitation
        Field 5: Ground Temp
        """
        payload = {
            "api_key": self.api_key,
            "field1": temp,
            "field2": hum,
            "field3": pres,
            "field4": rain,
        }
        if gtemp is not None:
            payload["field5"] = gtemp

        try:
            print(f"  [Cloud] Syncing {temp}°C to ThingSpeak...")
            response = requests.post(self.base_url, data=payload, timeout=10)
            if response.status_code == 200:
                print(f"  [Cloud] Success. entry_id: {response.text}")
                return True
            else:
                print(f"  [Cloud] Failed. Status: {response.status_code}")
                return False
        except Exception as e:
            print(f"  [Cloud] Error: {e}")
            return False

if __name__ == "__main__":
    # Test script
    sync = WeatherXCloud()
    sync.upload_seed_metrics(temp=36.0, hum=26.0, pres=1010.0, rain=0.0, gtemp=27.0)
