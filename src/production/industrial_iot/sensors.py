"""
Industrial Sensors Module
===========================

Industrial sensor interface with multi-protocol support.

Supports:
- Modbus TCP/RTU (registers)
- OPC UA (nodes)
- MQTT (pub/sub)

Classes:
    IndustrialSensor: Industrial sensor interface

Author: AI-Mastery-2026
"""

import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from .types import SensorReading, SensorType, ProtocolType

logger = logging.getLogger(__name__)


class IndustrialSensor:
    """
    Industrial sensor interface with multi-protocol support.

    Supports:
    - Modbus TCP/RTU (registers)
    - OPC UA (nodes)
    - MQTT (pub/sub)
    """

    def __init__(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        protocol: ProtocolType,
        address: str,
        sampling_rate_hz: float = 1.0,
    ):
        """
        Initialize sensor.

        Args:
            sensor_id: Unique identifier
            sensor_type: Type of sensor
            protocol: Communication protocol
            address: Modbus register / OPC node / MQTT topic
            sampling_rate_hz: Polling frequency
        """
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.protocol = protocol
        self.address = address
        self.sampling_rate_hz = sampling_rate_hz

        # Data buffer (circular for time-series)
        self.buffer: deque = deque(maxlen=1000)

        # Statistics
        self.reading_count = 0
        self.last_reading: Optional[SensorReading] = None

        # Calibration
        self.offset = 0.0
        self.scale = 1.0

        logger.info(f"Sensor initialized: {sensor_id} ({sensor_type.value})")

    def read(self) -> Optional[SensorReading]:
        """
        Read current sensor value.

        In production, this would use actual protocol libraries:
        - pymodbus for Modbus
        - opcua for OPC UA
        - paho-mqtt for MQTT
        """
        try:
            # Simulated read with realistic noise
            if self.sensor_type == SensorType.VIBRATION:
                raw_value = np.random.exponential(0.5) + 0.1
                unit = "mm/s"
            elif self.sensor_type == SensorType.TEMPERATURE:
                raw_value = 45 + np.random.randn() * 5
                unit = "°C"
            elif self.sensor_type == SensorType.PRESSURE:
                raw_value = 100 + np.random.randn() * 10
                unit = "bar"
            elif self.sensor_type == SensorType.FLOW:
                raw_value = 50 + np.random.randn() * 5
                unit = "m³/h"
            elif self.sensor_type == SensorType.CURRENT:
                raw_value = 100 + np.random.randn() * 10
                unit = "A"
            elif self.sensor_type == SensorType.ACOUSTIC:
                raw_value = 60 + np.random.randn() * 10
                unit = "dB"
            else:
                raw_value = np.random.random() * 100
                unit = "units"

            # Apply calibration
            value = raw_value * self.scale + self.offset

            reading = SensorReading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.now(),
                value=value,
                unit=unit,
            )

            self.buffer.append(reading)
            self.reading_count += 1
            self.last_reading = reading

            return reading

        except Exception as e:
            logger.error(f"Sensor read failed: {e}")
            return None

    def get_time_series(self, n_samples: int = 100) -> np.ndarray:
        """Get recent readings as numpy array."""
        readings = list(self.buffer)[-n_samples:]
        return np.array([r.value for r in readings])

    def get_latest_value(self) -> Optional[float]:
        """Get latest reading value."""
        if self.last_reading:
            return self.last_reading.value
        return None

    def calibrate(self, offset: float, scale: float):
        """Set calibration parameters."""
        self.offset = offset
        self.scale = scale
        logger.info(
            f"Sensor {self.sensor_id} calibrated: offset={offset}, scale={scale}"
        )

    def reset_stats(self):
        """Reset reading statistics."""
        self.reading_count = 0
        self.last_reading = None

    def get_stats(self) -> Dict[str, Any]:
        """Get sensor statistics."""
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type.value,
            "protocol": self.protocol.value,
            "reading_count": self.reading_count,
            "buffer_size": len(self.buffer),
            "last_value": self.last_reading.value if self.last_reading else None,
            "calibration": {"offset": self.offset, "scale": self.scale},
        }


class SensorArray:
    """Collection of industrial sensors for multi-sensor monitoring."""

    def __init__(self):
        self.sensors: Dict[str, IndustrialSensor] = {}

    def add_sensor(self, sensor: IndustrialSensor):
        """Add a sensor to the array."""
        self.sensors[sensor.sensor_id] = sensor
        logger.info(f"Added sensor to array: {sensor.sensor_id}")

    def read_all(self) -> Dict[str, float]:
        """Read all sensors and return values."""
        values = {}
        for sensor_id, sensor in self.sensors.items():
            reading = sensor.read()
            if reading:
                values[sensor_id] = reading.value
            else:
                values[sensor_id] = 0.0
        return values

    def get_feature_vector(self, sensor_ids: Optional[list] = None) -> np.ndarray:
        """Get sensor values as feature vector."""
        if sensor_ids is None:
            sensor_ids = sorted(self.sensors.keys())

        values = []
        for sid in sensor_ids:
            sensor = self.sensors.get(sid)
            if sensor and sensor.last_reading:
                values.append(sensor.last_reading.value)
            else:
                values.append(0.0)

        return np.array(values)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all sensors."""
        return {
            "total_sensors": len(self.sensors),
            "sensors": {sid: s.get_stats() for sid, s in self.sensors.items()},
        }
