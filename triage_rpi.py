#!/usr/bin/env python3
"""
Project 05 - Maintenance Triage Assistant
Raspberry Pi / standard Python 3

Modes
  live   : reads a real MPU6050 (+ optional DS18B20), computes vibration
           features, classifies NORMAL / OBSERVE / INSPECT, drives LEDs/buzzer,
           publishes JSON over MQTT.   (needs no CSV files)
  replay : same behaviour as the Wokwi version, replaying TEST rows from
           features.csv / baseline.csv / thresholds.csv.

Usage
  python3 triage_rpi.py                 # live mode (calibrates on first run)
  python3 triage_rpi.py --calibrate     # force a new calibration (healthy motor!)
  python3 triage_rpi.py --mode replay   # dataset replay
  python3 triage_rpi.py --mode replay --evaluate   # print false-alarm stats first

Install
  sudo apt install -y i2c-tools
  python3 -m venv --system-site-packages venv && source venv/bin/activate
  pip install smbus2 paho-mqtt        # gpiozero already ships with Raspberry Pi OS
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import struct
import threading
import time
import traceback
from collections import deque

import paho.mqtt.client as mqtt
from gpiozero import LED, Buzzer

# ================= CONFIG =================

MQTT_HOST = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_ROOT = "muhammeedd1/triage"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_FILE = os.path.join(BASE_DIR, "features.csv")
BASELINE_FILE = os.path.join(BASE_DIR, "baseline.csv")
THRESHOLDS_FILE = os.path.join(BASE_DIR, "thresholds.csv")
LIVE_CAL_FILE = os.path.join(BASE_DIR, "live_calibration.json")

# GPIO numbers are BCM (GPIO numbers, NOT physical pin numbers)
PIN_LED_GREEN = 27
PIN_LED_RED = 13
PIN_BUZZER = 23

# ---- replay ----
REPLAY_INTERVAL_S = 1.0
RANDOM_REPLAY = True

# ---- live / MPU6050 ----
I2C_BUS = 1
MPU_ADDR = 0x68              # 0x69 if AD0 is wired to 3.3V
ACCEL_LSB_PER_G = 8192.0     # +/-4 g range
SAMPLE_RATE_HZ = 1000        # MPU6050 accel output rate is 1 kHz max
WINDOW_SIZE = 512            # samples per measurement (~0.5 s)

LIVE_ASSET_ID = "motor1"
LIVE_SENSOR = "mpu6050"
LIVE_RPM = None              # put the nominal motor RPM here if you know it

# ---- live calibration (healthy motor) ----
CAL_WINDOWS = 40             # ~20 s
MIN_OBSERVE_RATIO = 1.30     # starting values, tune on your motor
MIN_INSPECT_RATIO = 2.00

# ==========================================

baselines = {}
thresholds = {}
sequence_number = 0

green_led = LED(PIN_LED_GREEN)
red_led = LED(PIN_LED_RED)
buzzer = Buzzer(PIN_BUZZER)       # active buzzer

mqtt_client = None
bus = None
latest_temp = None


# ---------- Utilities ----------

def number_or_none(value):
    try:
        value = str(value).strip()
        if not value or value.lower() in ("nan", "none", "null"):
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def label_or_none(value):
    number = number_or_none(value)
    return None if number is None else int(number)


def group_key(asset_id, sensor):
    return (str(asset_id).strip(), str(sensor).strip())


def read_csv(path):
    """Yield each CSV row as a dict with lower-case header names."""
    with open(path, newline="") as file:
        reader = csv.reader(file)
        try:
            header = [h.strip().strip('"').lower() for h in next(reader)]
        except StopIteration:
            raise ValueError(os.path.basename(path) + " is empty")

        for values in reader:
            if not values or len(values) < len(header):
                continue
            yield dict(zip(header, (v.strip().strip('"') for v in values)))


# ---------- Baselines and thresholds (replay mode) ----------

def load_baselines():
    baselines.clear()

    for row in read_csv(BASELINE_FILE):
        base = number_or_none(row.get("baseline_rms"))
        if base is not None and base > 0:
            key = group_key(row.get("asset_id", ""), row.get("sensor", ""))
            baselines[key] = {"baseline_rms": base}

    if not baselines:
        raise ValueError(
            "No valid baselines (need columns asset_id, sensor, baseline_rms)"
        )
    print("Baselines loaded:", len(baselines))


def load_thresholds():
    thresholds.clear()
    print("\nLoading precomputed thresholds...")

    for row in read_csv(THRESHOLDS_FILE):
        key = group_key(row.get("asset_id", ""), row.get("sensor", ""))
        observe = number_or_none(row.get("observe_ratio"))
        inspect = number_or_none(row.get("inspect_ratio"))
        count = number_or_none(row.get("normal_count"))

        if (observe is None or inspect is None or count is None
                or observe < 0 or inspect < observe or count < 20):
            print("Skipping invalid threshold:", key)
            continue

        thresholds[key] = {
            "observe": observe,
            "inspect": inspect,
            "normal_samples": int(count),
        }

    if not thresholds:
        raise ValueError("No valid thresholds loaded")

    print("Threshold groups loaded:", len(thresholds))
    for key in sorted(thresholds):
        item = thresholds[key]
        print("Asset:", key[0], "| Sensor:", key[1],
              "| N:", item["normal_samples"],
              "| OBSERVE >=", round(item["observe"], 4),
              "| INSPECT >=", round(item["inspect"], 4))


# ---------- Validation and classification ----------

def validate_measurement(row):
    for field in ("rms", "peak_to_peak", "kurtosis",
                  "crest_factor", "sampling_rate"):
        if number_or_none(row.get(field)) is None:
            return False, "Missing/invalid " + field

    if number_or_none(row["rms"]) < 0:
        return False, "Negative RMS"
    if number_or_none(row["peak_to_peak"]) < 0:
        return False, "Negative peak-to-peak"
    if number_or_none(row["sampling_rate"]) <= 0:
        return False, "Invalid sampling rate"

    # RPM is optional (the live sensor does not measure it)
    rpm = number_or_none(row.get("rpm"))
    if rpm is not None and rpm <= 0:
        return False, "Invalid RPM"

    return True, ""


def _no_decision(status, reason):
    return {"status": status, "valid": False, "reason": reason,
            "rms_ratio": None, "observe_threshold": None,
            "inspect_threshold": None}


def classify_row(row):
    # The ground-truth label is never used for the decision.
    valid, reason = validate_measurement(row)
    if not valid:
        return _no_decision("INVALID_DATA", reason)

    key = group_key(row.get("asset_id", ""), row.get("sensor", ""))

    if key not in baselines:
        return _no_decision("NO_BASELINE", "No baseline for asset/sensor")
    if key not in thresholds:
        return _no_decision("NO_THRESHOLD", "No threshold for asset/sensor")

    ratio = number_or_none(row["rms"]) / baselines[key]["baseline_rms"]
    limits = thresholds[key]

    if ratio >= limits["inspect"]:
        status = "INSPECT"
    elif ratio >= limits["observe"]:
        status = "OBSERVE"
    else:
        status = "NORMAL"

    return {"status": status, "valid": True, "reason": "",
            "rms_ratio": ratio,
            "observe_threshold": limits["observe"],
            "inspect_threshold": limits["inspect"]}


# ---------- Test evaluation (replay mode) ----------

def evaluate_test_split(rows):
    normal_total = normal_alerts = 0
    fault_total = fault_detected = fault_inspect = 0
    excluded = unavailable = 0

    print("\nEvaluating TEST split...")

    for row in rows:
        label = label_or_none(row.get("label"))
        if label is None or label == -1:
            excluded += 1
            continue

        result = classify_row(row)
        if not result["valid"]:
            unavailable += 1
            continue

        status = result["status"]
        if label == 0:
            normal_total += 1
            if status in ("OBSERVE", "INSPECT"):
                normal_alerts += 1
        elif label in (1, 2, 3):
            fault_total += 1
            if status in ("OBSERVE", "INSPECT"):
                fault_detected += 1
            if status == "INSPECT":
                fault_inspect += 1

    print("\n===== TEST SPLIT EVALUATION =====")
    print("Excluded labels:", excluded, "| Unavailable:", unavailable)

    if normal_total:
        print("Normal cases:", normal_total, "| False alarms:", normal_alerts,
              "| Rate:", round(100 * normal_alerts / normal_total, 2), "%")
    if fault_total:
        print("Fault cases:", fault_total, "| Detected:", fault_detected,
              "| Rate:", round(100 * fault_detected / fault_total, 2), "%")
        print("INSPECT:", fault_inspect,
              "| Rate:", round(100 * fault_inspect / fault_total, 2), "%")
    print("=================================\n")


# ---------- LEDs and buzzer ----------

def indicate_status(status):
    buzzer.off()

    if status == "NORMAL":
        green_led.on()
        red_led.off()
    else:
        green_led.off()
        red_led.on()

        if status == "INSPECT":
            buzzer.on()
            time.sleep(0.12)
            buzzer.off()


# ---------- MQTT ----------

def start_mqtt():
    global mqtt_client

    client_id = "triage-rpi-%06x" % random.getrandbits(24)
    try:                                   # paho-mqtt 2.x
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                             client_id=client_id)
    except AttributeError:                 # paho-mqtt 1.x
        client = mqtt.Client(client_id=client_id)

    client.reconnect_delay_set(min_delay=1, max_delay=30)
    client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_start()                    # background thread, auto-reconnect
    mqtt_client = client
    print("MQTT connecting to", MQTT_HOST, "...")


def publish_json(topic, payload):
    if mqtt_client is None or not mqtt_client.is_connected():
        return False
    info = mqtt_client.publish(topic, json.dumps(payload))
    return info.rc == mqtt.MQTT_ERR_SUCCESS


# ---------- Publishing ----------

def publish_row(row, source, unit="unknown", extra=None):
    global sequence_number
    sequence_number += 1

    asset_id = row.get("asset_id", "")
    sensor = row.get("sensor", "")
    device_id = "asset-%s-%s-muhammeedd1" % (asset_id, sensor)

    result = classify_row(row)
    indicate_status(result["status"])

    payload = {
        "device_id": device_id,
        "asset_id": asset_id,
        "sensor": sensor,
        "signal_index": row.get("signal_index"),
        "timestamp_ms": int(time.time() * 1000),
        "source": source,
        "unit": unit,
        "sampling_rate_hz": number_or_none(row.get("sampling_rate")),
        "rpm": number_or_none(row.get("rpm")),
        "features": {
            "rms": number_or_none(row.get("rms")),
            "peak_to_peak": number_or_none(row.get("peak_to_peak")),
            "kurtosis": number_or_none(row.get("kurtosis")),
            "crest_factor": number_or_none(row.get("crest_factor")),
        },
        "valid": result["valid"],
        "triage_preview": {
            "status": result["status"],
            "rms_ratio": result["rms_ratio"],
            "observe_threshold": result["observe_threshold"],
            "inspect_threshold": result["inspect_threshold"],
            "reason": result["reason"],
        },
        "ground_truth_label": label_or_none(row.get("label")),
        "split": row.get("split"),
    }
    if extra:
        payload.update(extra)

    ratio = result["rms_ratio"]
    print("Row:", sequence_number,
          "| RMS:", row.get("rms") if source != "live_mpu6050"
          else round(number_or_none(row.get("rms")), 5),
          "| Ratio:", round(ratio, 3) if ratio is not None else "N/A",
          "| Temp:", (extra or {}).get("temperature_c", "-"),
          "| Decision:", result["status"])

    base = TOPIC_ROOT + "/" + device_id
    publish_json(base + "/features", payload)
    publish_json(base + "/status", {
        "device_id": device_id,
        "state": "online",
        "source": source,
        "timestamp_ms": int(time.time() * 1000),
    })


# ---------- MPU6050 ----------

def init_mpu():
    global bus
    from smbus2 import SMBus

    bus = SMBus(I2C_BUS)

    who = bus.read_byte_data(MPU_ADDR, 0x75)
    if who != 0x68:
        print("Warning: WHO_AM_I = 0x%02X (clone chip?), continuing" % who)

    bus.write_byte_data(MPU_ADDR, 0x6B, 0x01)   # wake up, PLL clock
    time.sleep(0.1)
    bus.write_byte_data(MPU_ADDR, 0x19, 0x00)   # sample-rate divider = 0
    bus.write_byte_data(MPU_ADDR, 0x1A, 0x00)   # DLPF off (260 Hz accel BW)
    bus.write_byte_data(MPU_ADDR, 0x1C, 0x08)   # accel range +/-4 g
    print("MPU6050 ready")


def read_accel_magnitude():
    raw = bus.read_i2c_block_data(MPU_ADDR, 0x3B, 6)
    ax, ay, az = struct.unpack(">hhh", bytes(raw))
    ax /= ACCEL_LSB_PER_G
    ay /= ACCEL_LSB_PER_G
    az /= ACCEL_LSB_PER_G
    return math.sqrt(ax * ax + ay * ay + az * az)


def wait_until(t):
    while True:
        remaining = t - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > 0.0005:
            time.sleep(remaining - 0.0004)


def sample_window():
    """Collect WINDOW_SIZE samples; return (samples, real_sampling_rate)."""
    period = 1.0 / SAMPLE_RATE_HZ
    samples = []
    start = time.perf_counter()
    next_t = start

    for _ in range(WINDOW_SIZE):
        samples.append(read_accel_magnitude())
        next_t += period
        wait_until(next_t)

    elapsed = time.perf_counter() - start
    return samples, WINDOW_SIZE / elapsed


def compute_features(samples):
    n = len(samples)
    mean = sum(samples) / n                 # removes gravity (DC)
    ac = [s - mean for s in samples]

    var = sum(x * x for x in ac) / n
    rms = math.sqrt(var)
    peak_to_peak = max(samples) - min(samples)
    peak = max(abs(x) for x in ac)
    crest = peak / rms if rms > 0 else 0.0
    # Pearson kurtosis (a normal signal gives about 3)
    kurt = (sum(x ** 4 for x in ac) / n) / (var * var) if var > 0 else 0.0

    return rms, peak_to_peak, kurt, crest


# ---------- DS18B20 (1-Wire, optional) ----------

def temperature_worker(path):
    global latest_temp
    while True:
        try:
            with open(path) as file:
                lines = file.read().splitlines()
            if len(lines) >= 2 and lines[0].strip().endswith("YES"):
                latest_temp = round(int(lines[1].split("t=")[1]) / 1000.0, 1)
        except Exception:
            latest_temp = None
        time.sleep(2)


def start_temperature():
    paths = glob.glob("/sys/bus/w1/devices/28-*/w1_slave")
    if not paths:
        print("DS18B20 not found - temperature disabled")
        return
    threading.Thread(target=temperature_worker, args=(paths[0],),
                     daemon=True).start()
    print("DS18B20 found:", paths[0].split("/")[-2])


# ---------- Live calibration ----------

def calibrate():
    print("\nCALIBRATION: run the motor in HEALTHY condition and keep it steady")
    print("Collecting %d windows..." % CAL_WINDOWS)

    values = []
    for i in range(CAL_WINDOWS):
        samples, fs = sample_window()
        rms = compute_features(samples)[0]
        values.append(rms)
        print("  window %2d/%d | RMS %.5f g | fs %.0f Hz"
              % (i + 1, CAL_WINDOWS, rms, fs))

    baseline = sum(values) / len(values)
    if baseline <= 0:
        raise ValueError("Baseline RMS is zero - is the sensor connected?")

    ratios = [v / baseline for v in values]
    std = math.sqrt(sum((r - 1.0) ** 2 for r in ratios) / len(ratios))

    cal = {
        "baseline_rms": baseline,
        "observe_ratio": max(1.0 + 3 * std, MIN_OBSERVE_RATIO),
        "inspect_ratio": max(1.0 + 6 * std, MIN_INSPECT_RATIO),
        "normal_count": len(values),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(LIVE_CAL_FILE, "w") as file:
        json.dump(cal, file, indent=2)

    print("Calibration saved:", cal)
    return cal


def load_live_calibration():
    if not os.path.exists(LIVE_CAL_FILE):
        return None
    with open(LIVE_CAL_FILE) as file:
        return json.load(file)


# ---------- Run modes ----------

def run_live(args):
    print("\nMaintenance Triage Assistant - LIVE (MPU6050)")
    init_mpu()

    cal = None if args.calibrate else load_live_calibration()
    if cal is None:
        cal = calibrate()
    else:
        print("Using saved calibration from", cal.get("created"))

    key = group_key(LIVE_ASSET_ID, LIVE_SENSOR)
    baselines[key] = {"baseline_rms": cal["baseline_rms"]}
    thresholds[key] = {"observe": cal["observe_ratio"],
                       "inspect": cal["inspect_ratio"],
                       "normal_samples": cal["normal_count"]}
    print("OBSERVE >= %.3f | INSPECT >= %.3f (RMS ratio)"
          % (cal["observe_ratio"], cal["inspect_ratio"]))

    start_temperature()
    start_mqtt()

    while True:
        samples, fs = sample_window()
        rms, p2p, kurt, crest = compute_features(samples)

        row = {
            "asset_id": LIVE_ASSET_ID,
            "sensor": LIVE_SENSOR,
            "signal_index": sequence_number + 1,
            "rms": rms,
            "peak_to_peak": p2p,
            "kurtosis": kurt,
            "crest_factor": crest,
            "rpm": LIVE_RPM if LIVE_RPM is not None else "",
            "sampling_rate": fs,
            "label": "",
            "split": "live",
        }

        publish_row(row, "live_mpu6050", unit="g",
                    extra={"temperature_c": latest_temp})


def run_replay(args):
    print("\nMaintenance Triage Assistant - DATASET REPLAY")
    load_baselines()
    load_thresholds()

    rows = [r for r in read_csv(FEATURES_FILE)
            if r.get("split", "").lower() == "test"]
    if not rows:
        raise ValueError("No test rows found in features.csv")
    print("Test rows loaded:", len(rows))

    if args.evaluate:
        evaluate_test_split(rows)

    start_mqtt()
    recent = deque(maxlen=min(60, len(rows) // 2))

    while True:
        if RANDOM_REPLAY:
            for _ in range(50):
                index = random.randrange(len(rows))
                if index not in recent:
                    break
            recent.append(index)
            publish_row(rows[index], "dataset_replay")
            time.sleep(REPLAY_INTERVAL_S)
        else:
            for row in rows:
                publish_row(row, "dataset_replay")
                time.sleep(REPLAY_INTERVAL_S)
            print("Test replay completed; restarting...")


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("live", "replay"), default="live")
    parser.add_argument("--calibrate", action="store_true",
                        help="live mode: redo calibration on a healthy motor")
    parser.add_argument("--evaluate", action="store_true",
                        help="replay mode: print false-alarm/detection stats")
    args = parser.parse_args()

    try:
        if args.mode == "live":
            run_live(args)
        else:
            run_replay(args)
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as exc:
        print("FATAL ERROR:", exc)
        traceback.print_exc()
    finally:
        green_led.off()
        red_led.off()
        buzzer.off()
        if mqtt_client is not None:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        if bus is not None:
            bus.close()


if __name__ == "__main__":
    main()
