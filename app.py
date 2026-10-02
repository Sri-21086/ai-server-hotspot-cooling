import json
import time
import zipfile
import threading
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
import paho.mqtt.client as mqtt
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Server Hotspot Cooling System",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ADVANCED DARK UI
# ============================================================

st.markdown(
    """
<style>
.stApp {
    background:
        radial-gradient(circle at 15% 5%, rgba(0, 212, 255, 0.08), transparent 28%),
        radial-gradient(circle at 90% 15%, rgba(75, 55, 255, 0.10), transparent 30%),
        #070b12;
    color: #f5f7fb;
}

.block-container {
    max-width: 1800px;
    padding-top: 1.0rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

[data-testid="stSidebar"] {
    background: #0b1018;
    border-right: 1px solid #1e2a3a;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #111a27, #0d141f);
    border: 1px solid #243247;
    border-radius: 14px;
    padding: 12px;
}

[data-testid="stMetricValue"] {
    color: #f8fbff;
}

.server-card {
    background:
        linear-gradient(145deg, rgba(20, 35, 58, 0.98), rgba(10, 17, 29, 0.98));
    border: 1px solid #263a56;
    border-radius: 18px;
    padding: 18px;
    min-height: 650px;
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.28);
    margin-bottom: 18px;
}

.server-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
}

.server-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #ffffff;
}

.server-rack {
    color: #8998aa;
    font-size: 0.82rem;
    margin-top: 3px;
}

.badge {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 800;
    border: 1px solid rgba(255,255,255,0.12);
}

.badge-safe { color: #68f3a1; background: rgba(40, 190, 100, 0.12); }
.badge-warm { color: #ffd166; background: rgba(255, 190, 50, 0.12); }
.badge-hot { color: #ff9b55; background: rgba(255, 120, 40, 0.13); }
.badge-critical { color: #ff6969; background: rgba(255, 55, 55, 0.13); }
.badge-blue { color: #6bdcff; background: rgba(0, 190, 255, 0.12); }

.temp-panel {
    margin-top: 16px;
    padding: 16px;
    border-radius: 16px;
    background: linear-gradient(145deg, #101c2c, #0a111d);
    border: 1px solid #263c59;
}

.temp-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}

.temp-label {
    color: #b8c4d3;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
}

.temp-value {
    color: #ffffff;
    font-size: 2.0rem;
    font-weight: 900;
    margin-top: 5px;
}

.pred-value {
    color: #76e7ff;
    font-size: 1.65rem;
    font-weight: 900;
    margin-top: 5px;
}

.delta {
    color: #ff6b9c;
    font-size: 0.78rem;
    font-weight: 700;
    margin-top: 4px;
}

.gauge {
    margin-top: 14px;
    height: 12px;
    border-radius: 99px;
    background: linear-gradient(
        90deg,
        #35d77f 0%,
        #35d77f 50%,
        #ffd166 50%,
        #ffd166 66.67%,
        #ff8c42 66.67%,
        #ff8c42 83.33%,
        #ff4f5e 83.33%,
        #ff4f5e 100%
    );
    position: relative;
    overflow: hidden;
}

.gauge-marker {
    position: absolute;
    top: -4px;
    width: 4px;
    height: 20px;
    background: white;
    border-radius: 5px;
    box-shadow: 0 0 10px rgba(255,255,255,0.7);
}

.gauge-labels {
    display: flex;
    justify-content: space-between;
    color: #7f8da0;
    font-size: 0.68rem;
    margin-top: 4px;
}

.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 16px;
}

.info-box {
    background: rgba(15, 25, 40, 0.85);
    border: 1px solid #26384f;
    border-radius: 12px;
    padding: 11px;
}

.info-label {
    color: #7f8da0;
    font-size: 0.72rem;
    font-weight: 700;
}

.info-value {
    color: #eef5ff;
    font-weight: 800;
    margin-top: 3px;
}

.formula-box {
    margin-top: 15px;
    padding: 10px 12px;
    background: #09111d;
    border: 1px solid #20334d;
    border-radius: 10px;
    color: #a9c4dd;
    font-size: 0.72rem;
    overflow-x: auto;
}

.live-dot {
    color: #55ef94;
    font-weight: 800;
}

.off-dot {
    color: #8d99a8;
    font-weight: 800;
}

.hero {
    background: linear-gradient(110deg, rgba(17, 34, 57, 0.95), rgba(8, 15, 26, 0.95));
    border: 1px solid #223752;
    border-radius: 20px;
    padding: 22px 25px;
    margin-bottom: 18px;
}

.hero-title {
    font-size: 1.55rem;
    font-weight: 900;
    color: #ffffff;
}

.hero-subtitle {
    color: #93a5b8;
    margin-top: 4px;
}

.status-pill {
    display: inline-block;
    margin-top: 10px;
    padding: 7px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 800;
    background: rgba(0, 205, 255, 0.10);
    color: #6de4ff;
    border: 1px solid rgba(0, 205, 255, 0.25);
}

.small-note {
    color: #8d9bad;
    font-size: 0.82rem;
}

.section-title {
    font-size: 1.15rem;
    font-weight: 900;
    margin: 18px 0 10px 0;
}

.disconnect-panel {
    background: linear-gradient(135deg, rgba(35, 44, 58, 0.96), rgba(17, 23, 32, 0.96));
    border: 1px solid rgba(150, 165, 185, 0.18);
    border-radius: 18px;
    padding: 38px;
    margin-top: 24px;
    text-align: center;
}

.disconnect-title {
    color: #dbe4ef;
    font-size: 1.5rem;
    font-weight: 900;
}

.disconnect-text {
    color: #91a0b2;
    margin-top: 9px;
    font-size: 0.95rem;
    line-height: 1.6;
}

div[data-testid="stButton"] > button {
    border-radius: 12px;
    font-weight: 800;
}

@media (max-width: 900px) {
    .server-card {
        min-height: auto;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ZIP_FILE = BASE_DIR / "server_cooling_5_models_1000_rows.zip"
DATA_DIR = BASE_DIR / "_model_data"

MQTT_BROKER = "broker.hivemq.com"
MQTT_TCP_PORT = 1883
MQTT_WEBSOCKET_PORT = 8000
MQTT_TOPIC = "ai-cooling/WOKWI-ESP32-001"

SAFE_TEMP = 35.0
WARNING_TEMP = 40.0
CRITICAL_TEMP = 50.0

CSV_NAMES = [
    "server_cooling_model1_1000_rows.csv",
    "server_cooling_model2_1000_rows.csv",
    "server_cooling_model3_1000_rows.csv",
    "server_cooling_model4_1000_rows.csv",
    "server_cooling_model5_1000_rows.csv",
]


# ============================================================
# DATASET PREPARATION
# ============================================================

def prepare_datasets():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    found_files = {file.name: file for file in DATA_DIR.rglob("*.csv")}

    if all(name in found_files for name in CSV_NAMES):
        return [found_files[name] for name in CSV_NAMES]

    if not ZIP_FILE.exists():
        raise FileNotFoundError(
            "Dataset ZIP file was not found. Expected: "
            f"{ZIP_FILE.name}"
        )

    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    found_files = {file.name: file for file in DATA_DIR.rglob("*.csv")}

    missing = [name for name in CSV_NAMES if name not in found_files]

    if missing:
        raise FileNotFoundError(
            "These CSV files were not found inside the ZIP:\n"
            + "\n".join(missing)
        )

    return [found_files[name] for name in CSV_NAMES]


@st.cache_data
def load_datasets():
    files = prepare_datasets()

    frames = [pd.read_csv(file) for file in files]

    for df in frames:
        df.columns = df.columns.astype(str).str.strip()

    return tuple(frames)


# ============================================================
# MQTT STATE
# ============================================================

class MQTTState:
    """Persistent MQTT listener that stays alive across Streamlit reruns."""

    def __init__(self):
        self.lock = threading.RLock()
        self.connected = False
        self.connecting = False
        self.desired_connection = False
        self.latest_data = None
        self.last_update = None
        self.message_count = 0
        self.connection_error = None
        self.last_payload = ""
        self.stop_event = threading.Event()
        self.worker_thread = None

        client_id = "STREAMLIT-" + uuid.uuid4().hex[:12]

        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            transport="websockets",
        )
        self.client.ws_set_options(path="/mqtt")
        self.client.reconnect_delay_set(min_delay=2, max_delay=15)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

    def on_connect(self, client, userdata, flags, reason_code, properties):
        print("MQTT connection result:", reason_code)
        if reason_code == 0:
            with self.lock:
                self.connected = True
                self.connecting = False
                self.connection_error = None

            result = client.subscribe(MQTT_TOPIC, qos=0)
            print("Subscribed:", MQTT_TOPIC, result)
        else:
            with self.lock:
                self.connected = False
                self.connecting = False
                self.connection_error = f"MQTT connection refused: {reason_code}"

    def on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8", errors="replace")
            data = json.loads(payload)
            servers = data.get("servers")

            if not isinstance(servers, list) or not servers:
                return

            received_at = time.time()
            with self.lock:
                self.latest_data = data
                self.last_payload = payload
                self.last_update = received_at
                self.message_count += 1
                self.connected = True
                self.connection_error = None

            print(
                f"MQTT LIVE MESSAGE #{self.message_count}: "
                f"{len(servers)} servers received"
            )
        except Exception as exc:
            print("MQTT message processing error:", exc)

    def on_disconnect(
        self,
        client,
        userdata,
        disconnect_flags,
        reason_code,
        properties,
    ):
        with self.lock:
            self.connected = False
            if self.desired_connection and not self.stop_event.is_set():
                self.connecting = True

        print("MQTT disconnected:", reason_code)

    def _worker(self):
        """Keep the MQTT connection alive independently of Streamlit reruns."""
        while not self.stop_event.is_set():
            with self.lock:
                desired = self.desired_connection

            if not desired:
                time.sleep(0.5)
                continue

            try:
                with self.lock:
                    self.connecting = True
                    self.connection_error = None

                # loop_forever() maintains the network loop and handles
                # normal reconnects. It returns when the client disconnects.
                self.client.connect(
                    MQTT_BROKER,
                    MQTT_WEBSOCKET_PORT,
                    keepalive=60,
                )
                self.client.loop_forever(retry_first_connection=True)

            except Exception as exc:
                with self.lock:
                    self.connected = False
                    self.connecting = False
                    self.connection_error = str(exc)

                # Do not hammer the public broker after a transient failure.
                time.sleep(2)

            with self.lock:
                should_continue = self.desired_connection

            if should_continue and not self.stop_event.is_set():
                time.sleep(0.5)

        with self.lock:
            self.connected = False
            self.connecting = False

    def connect(self):
        with self.lock:
            self.desired_connection = True
            self.stop_event.clear()

            if self.worker_thread is not None and self.worker_thread.is_alive():
                return True

            self.worker_thread = threading.Thread(
                target=self._worker,
                name="mqtt-background-listener",
                daemon=True,
            )
            self.worker_thread.start()

        return True

    def disconnect(self):
        """Stop MQTT completely and clear all live telemetry."""
        with self.lock:
            self.desired_connection = False
            self.stop_event.set()
            self.connected = False
            self.connecting = False
            self.latest_data = None
            self.last_update = None
            self.last_payload = ""
            self.message_count = 0
            self.connection_error = None

        try:
            self.client.disconnect()
        except Exception:
            pass

        thread = self.worker_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

        with self.lock:
            self.worker_thread = None

    def get_data(self):
        with self.lock:
            if self.latest_data is None:
                return None
            return json.loads(json.dumps(self.latest_data))

    def is_connected(self):
        with self.lock:
            return self.connected

    def is_connecting(self):
        with self.lock:
            return self.connecting

    def get_last_update(self):
        with self.lock:
            return self.last_update

    def get_message_count(self):
        with self.lock:
            return self.message_count

    def get_error(self):
        with self.lock:
            return self.connection_error

    def get_last_payload(self):
        with self.lock:
            return self.last_payload


@st.cache_resource
def get_mqtt_state(cache_version="v4"):
    # Changing cache_version forces Streamlit Cloud to create a fresh MQTTState
    # instead of reusing an object created by an older app.py version.
    return MQTTState()


mqtt_state = get_mqtt_state("v4")


def mqtt_is_connecting(state):
    """Compatibility helper for cached MQTTState objects from older deployments."""
    method = getattr(state, "is_connecting", None)
    if callable(method):
        try:
            return bool(method())
        except Exception:
            pass
    return bool(getattr(state, "connecting", False))
# ------------------------------------------------------------
# MQTT OPERATION STATE
# ------------------------------------------------------------
# CONNECT / DISCONNECT controls the complete live operation.
# The first page load starts in a stopped/disconnected state.
st.session_state.setdefault("mqtt_enabled", False)

if st.session_state["mqtt_enabled"]:
    mqtt_state.connect()


# ============================================================
# MODEL TRAINING
# ============================================================

# These are the feature columns used by the current datasets/app.
# The dashboard separately exposes the original 11 simulated ML
# parameters in build_features().
MODEL_SPECS = {
    "model1": {
        "features": [
            "Ambient Temperature C",
            "Current GPU Temperature C",
            "CPU Temperature C",
            "Server Load Percent",
            "Humidity Percent",
            "Current Fan Speed Percent",
        ],
        "target": "Future Temperature C",
        "kind": "regression",
        "name": "Future GPU Temperature",
    },
    "model2": {
        "features": [
            "Ambient Temperature C",
            "Current GPU Temperature C",
            "CPU Temperature C",
            "Server Load Percent",
            "Humidity Percent",
            "Current Fan Speed Percent",
        ],
        "target": "Required Fan Speed Percent",
        "kind": "regression",
        "name": "Required Fan Speed",
    },
    "model3": {
        "features": [
            "Ambient Temperature C",
            "Current GPU Temperature C",
            "CPU Temperature C",
            "Server Load Percent",
            "Current Fan Speed Percent",
            "Temperature Rise C",
        ],
        "target": "Hotspot Risk",
        "kind": "classification",
        "name": "Hotspot Risk",
    },
    "model4": {
        "features": [
            "Ambient Temperature C",
            "Current GPU Temperature C",
            "CPU Temperature C",
            "Server Load Percent",
            "Fan Speed Percent",
            "Temperature Before Cooling C",
        ],
        "target": "Cooling Effectiveness Percent",
        "kind": "regression",
        "name": "Cooling Effectiveness",
    },
    "model5": {
        "features": [
            "Ambient Temperature C",
            "Current GPU Temperature C",
            "CPU Temperature C",
            "Server Load Percent",
            "Fan Speed Percent",
            "Hotspot Risk",
        ],
        "target": "Overheating Warning",
        "kind": "classification",
        "name": "Overheating Warning",
    },
}


def train_one_regressor(df, features, target):
    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target], errors="coerce")

    valid = X.notna().all(axis=1) & y.notna()

    X = X.loc[valid]
    y = y.loc[valid]

    if len(X) < 10:
        raise ValueError(f"Not enough valid rows for target '{target}'.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=180,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)

    metrics = {
        "R2": float(r2_score(y_test, pred)),
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(mse ** 0.5),
        "rows": int(len(X)),
    }

    return model, metrics


def train_classifier(df, features, target):
    # Some of the supplied classification datasets use text labels
    # such as Low/Medium/High and Normal/Warning/Critical.  Encode
    # those labels explicitly instead of trying to convert them to NaN.
    X = df[features].copy()
    y = df[target].astype(str).str.strip()

    feature_maps = {}

    for feature in features:
        if pd.api.types.is_numeric_dtype(X[feature]):
            X[feature] = pd.to_numeric(X[feature], errors="coerce")
        else:
            values = X[feature].astype(str).str.strip()
            classes = sorted(values.dropna().unique().tolist())
            mapping = {label: i for i, label in enumerate(classes)}
            feature_maps[feature] = mapping
            X[feature] = values.map(mapping)

    target_classes = sorted(y.dropna().unique().tolist())
    target_map = {label: i for i, label in enumerate(target_classes)}
    y_encoded = y.map(target_map)

    valid = X.notna().all(axis=1) & y_encoded.notna()
    X = X.loc[valid]
    y_encoded = y_encoded.loc[valid].astype(int)

    if len(X) < 10:
        raise ValueError(f"Not enough valid rows for target '{target}'.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded if y_encoded.nunique() > 1 else None,
    )

    model = RandomForestClassifier(
        n_estimators=180,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "Accuracy": float(accuracy_score(y_test, pred)),
        "rows": int(len(X)),
    }

    return model, metrics, feature_maps, target_classes


@st.cache_resource
def train_models():
    df1, df2, df3, df4, df5 = load_datasets()
    frames = [df1, df2, df3, df4, df5]

    trained = {}

    for index, (key, spec) in enumerate(MODEL_SPECS.items()):
        df = frames[index]

        missing = [
            col
            for col in spec["features"] + [spec["target"]]
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{key} dataset is missing required columns: "
                + ", ".join(missing)
            )

        feature_maps = {}
        target_classes = None

        if spec["kind"] == "classification":
            model, metrics, feature_maps, target_classes = train_classifier(
                df,
                spec["features"],
                spec["target"],
            )
        else:
            model, metrics = train_one_regressor(
                df,
                spec["features"],
                spec["target"],
            )

        trained[key] = {
            "model": model,
            "metrics": metrics,
            "features": spec["features"],
            "target": spec["target"],
            "kind": spec["kind"],
            "name": spec["name"],
            "feature_maps": feature_maps,
            "target_classes": target_classes,
        }

    return trained


# ============================================================
# ORIGINAL 11 ML PARAMETERS
# ============================================================

def build_features(server):
    """
    Temperature comes directly from Wokwi.
    The remaining parameters are estimated/simulated for the
    ML demonstration because the Wokwi model currently measures
    temperature directly.
    """

    temp = float(server.get("temperature", 22.0))
    cooling = bool(server.get("cooling_active", False))

    gpu_usage = min(100.0, max(10.0, 35.0 + (temp - 25.0) * 4.0))
    gpu_power = min(450.0, max(80.0, 120.0 + gpu_usage * 2.2))
    cpu_usage = min(100.0, max(15.0, gpu_usage * 0.72))
    cpu_temp = min(90.0, max(30.0, 35.0 + (temp - 25.0) * 0.65))

    ambient = 25.0
    inlet = max(18.0, temp - 8.0)
    outlet = temp + 5.0

    fan_speed = min(
        100.0,
        max(25.0, 40.0 + (temp - 30.0) * 3.0),
    )

    if cooling:
        fan_speed = min(100.0, fan_speed + 15.0)

    airflow = min(
        3.0,
        max(0.4, 0.8 + fan_speed / 100.0 * 1.5),
    )

    workload = min(
        100.0,
        max(10.0, gpu_usage * 0.92),
    )

    return pd.DataFrame(
        [
            {
                "GPU_usage_percent": gpu_usage,
                "GPU_power_W": gpu_power,
                "CPU_usage_percent": cpu_usage,
                "CPU_temperature_C": cpu_temp,
                "GPU_temperature_C": temp,
                "ambient_temperature_C": ambient,
                "inlet_temperature_C": inlet,
                "outlet_temperature_C": outlet,
                "fan_speed_percent": fan_speed,
                "airflow_m3_s": airflow,
                "workload_percent": workload,
            }
        ]
    )


def make_dataset_model_inputs(server, original):
    """
    Convert the live Wokwi/original 11-parameter representation
    into the exact columns required by the current five CSV datasets.
    """

    temp = float(server.get("temperature", original["GPU_temperature_C"]))
    previous = float(
        server.get(
            "previous_temperature",
            max(0.0, temp - 0.5),
        )
    )

    cooling = bool(server.get("cooling_active", False))

    gpu_usage = float(original["GPU_usage_percent"])
    cpu_usage = float(original["CPU_usage_percent"])
    cpu_temp = float(original["CPU_temperature_C"])
    ambient = float(original["ambient_temperature_C"])
    fan = float(original["fan_speed_percent"])
    humidity = 45.0

    server_load = float(
        server.get(
            "workload_percent",
            original["workload_percent"],
        )
    )

    temperature_rise = temp - previous

    common = {
        "Ambient Temperature C": ambient,
        "Current GPU Temperature C": temp,
        "CPU Temperature C": cpu_temp,
        "Server Load Percent": server_load,
        "Humidity Percent": humidity,
        "Current Fan Speed Percent": fan,
        "Fan Speed Percent": fan,
        "Temperature Rise C": temperature_rise,
        "Temperature Before Cooling C": previous,
        "Hotspot Risk": 100.0 if temp >= WARNING_TEMP else max(0.0, temperature_rise * 10.0),
        "GPU_usage_percent": gpu_usage,
        "CPU_usage_percent": cpu_usage,
        "cooling_active": cooling,
    }

    return common


def predict_server(server, models):
    original = build_features(server)
    original_row = original.iloc[0].to_dict()
    inputs = make_dataset_model_inputs(server, original_row)

    results = {}

    for key, bundle in models.items():
        row_values = {}

        for feature in bundle["features"]:
            value = inputs[feature]
            mapping = bundle.get("feature_maps", {}).get(feature)

            if mapping is not None:
                # Model 5 uses the dataset's Low/Medium/High hotspot labels.
                # Convert the live numeric risk into the same labels.
                if feature == "Hotspot Risk" and isinstance(value, (int, float)):
                    if float(value) < 33.0:
                        value = "Low"
                    elif float(value) < 66.0:
                        value = "Medium"
                    else:
                        value = "High"

                value = mapping.get(str(value).strip())

            row_values[feature] = value

        row = pd.DataFrame([row_values])
        value = bundle["model"].predict(row)[0]

        if bundle["kind"] == "classification":
            classes = bundle.get("target_classes") or []
            label = classes[int(value)] if int(value) < len(classes) else str(value)

            if key == "model3":
                risk_map = {
                    "Low": 20.0,
                    "Medium": 55.0,
                    "High": 90.0,
                }
                results[key] = risk_map.get(label, 50.0)
            elif key == "model5":
                results[key] = 0 if label == "Normal" else 1
            else:
                results[key] = int(value)
        else:
            results[key] = float(value)

    temp = float(server.get("temperature", 0.0))

    # Hardware temperature has priority for safety display.
    if temp >= CRITICAL_TEMP:
        results["model3"] = 100.0
        results["model5"] = 1

    elif temp >= WARNING_TEMP:
        results["model3"] = max(float(results["model3"]), 80.0)
        results["model5"] = 1

    elif temp >= SAFE_TEMP:
        results["model3"] = max(float(results["model3"]), 35.0)

    results["model3"] = max(0.0, min(100.0, float(results["model3"])))
    results["model2"] = max(0.0, min(100.0, float(results["model2"])))
    results["model4"] = max(0.0, min(100.0, float(results["model4"])))

    return results, original


# ============================================================
# STATUS HELPERS
# ============================================================

def temperature_status(temp):
    if temp >= CRITICAL_TEMP:
        return "CRITICAL"
    if temp >= WARNING_TEMP:
        return "HIGH TEMPERATURE"
    if temp >= SAFE_TEMP:
        return "WARM - MONITORING"
    return "NORMAL"


def status_badge(temp):
    if temp >= CRITICAL_TEMP:
        return "badge-critical"
    if temp >= WARNING_TEMP:
        return "badge-hot"
    if temp >= SAFE_TEMP:
        return "badge-warm"
    return "badge-safe"


def protection_status(temp):
    if temp >= CRITICAL_TEMP:
        return "CRITICAL PROTECTION"
    if temp >= WARNING_TEMP:
        return "ACTIVE"
    return "NORMAL"


def hotspot_status(server):
    temp = float(server.get("temperature", 0))
    predicted = float(server.get("predicted_temperature", temp))
    cooling = bool(server.get("cooling_active", False))

    if temp >= CRITICAL_TEMP:
        return "CRITICAL HOTSPOT"
    if temp >= WARNING_TEMP:
        return "HOTSPOT / HIGH THERMAL STRESS"
    if predicted >= SAFE_TEMP:
        return "HOTSPOT PREDICTED"
    if cooling:
        return "COOLING ACTIVE"
    return "NO HOTSPOT"


def thermal_stress(temp, previous, cooling):
    if temp >= CRITICAL_TEMP:
        return "CRITICAL"
    if temp >= WARNING_TEMP:
        return "HIGH"
    if temp >= SAFE_TEMP:
        return "MODERATE"
    if cooling:
        return "LOW / COOLING"
    return "LOW"


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ System Control")

    st.markdown("### MQTT Device")
    st.code("WOKWI-ESP32-001")

    st.write("Broker:", MQTT_BROKER)
    st.write("Wokwi MQTT Port:", MQTT_TCP_PORT)
    st.write("Dashboard WebSocket Port:", MQTT_WEBSOCKET_PORT)
    st.write("Topic:", MQTT_TOPIC)

    st.divider()

    # CONNECT starts the complete live operation.
    if st.button("🔌 CONNECT / RECONNECT", use_container_width=True):
        st.session_state["mqtt_enabled"] = True
        mqtt_state.connect()
        st.rerun()

    # DISCONNECT stops MQTT, clears live telemetry, and removes the
    # auto-refresh fragment on the following full rerun.
    if st.button("🔴 DISCONNECT", use_container_width=True):
        st.session_state["mqtt_enabled"] = False
        mqtt_state.disconnect()
        st.rerun()

    if not st.session_state["mqtt_enabled"]:
        st.info("⏸️ SYSTEM DISCONNECTED — live MQTT + auto-refresh stopped")
    elif mqtt_state.is_connected():
        st.success("🟢 MQTT CONNECTED — listening continuously")
    elif mqtt_is_connecting(mqtt_state):
        st.info("🔄 MQTT CONNECTING / RECONNECTING…")
    else:
        st.warning("🟡 MQTT NOT CONNECTED")

    st.divider()

    st.write("Servers: 4")
    st.write(
        "UI refresh: 1 second while connected"
        if st.session_state["mqtt_enabled"]
        else "UI refresh: STOPPED"
    )
    st.write(
        "MQTT listener: continuous while connected"
        if st.session_state["mqtt_enabled"]
        else "MQTT listener: STOPPED"
    )

    st.divider()

    st.metric(
        "MQTT Messages Received",
        mqtt_state.get_message_count(),
    )

    last_update = mqtt_state.get_last_update()

    if st.session_state["mqtt_enabled"] and last_update is not None:
        age = max(0.0, time.time() - last_update)
        st.caption(f"Last Wokwi message: {age:.1f}s ago")

    error = mqtt_state.get_error()

    if error and st.session_state["mqtt_enabled"]:
        st.error(error)

    st.divider()

    st.caption(
        "Wokwi directly measures temperature. "
        "Other ML parameters are estimated for the demonstration."
    )


# ============================================================
# TRAIN MODELS
# ============================================================

try:
    models = train_models()
    model_error = None
except Exception as exc:
    models = None
    model_error = str(exc)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">
    <div class="hero-title">🌡️ AI-Based Server Hotspot Cooling System</div>
    <div class="hero-subtitle">
        Data-Center GPU Thermal Telemetry & Predictive Dynamic Fan Actuation
    </div>
    <div class="status-pill">
        ESP32 + MQTT + Random Forest + Real-Time Wokwi Telemetry
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL ERROR
# ============================================================

if model_error:
    st.error("ML model initialization failed.")
    st.code(model_error)
    st.stop()


# ============================================================
# LIVE DASHBOARD
# ============================================================

@st.fragment(run_every=1)
def realtime_dashboard():

    # Safety guard for the transition immediately after DISCONNECT.
    # No MQTT/ML work is performed while the live operation is disabled.
    if not st.session_state.get("mqtt_enabled", False):
        return

    data = mqtt_state.get_data()

    # --------------------------------------------------------
    # No MQTT DATA YET
    # --------------------------------------------------------

    if data is None:
        st.info(
            "Waiting for live Wokwi data. Click CONNECT and keep Wokwi running."
        )
        return

    servers = data.get("servers", [])

    if not servers:
        st.warning("MQTT message received, but no server records were found.")
        return

    # --------------------------------------------------------
    # LIVE STATUS
    # --------------------------------------------------------

    last_update = mqtt_state.get_last_update()
    age = time.time() - last_update if last_update else None
    msg_count = mqtt_state.get_message_count()

    if mqtt_state.is_connected() and age is not None and age <= 10:
        live_text = (
            f"🟢 LIVE — MQTT is receiving Wokwi data continuously "
            f"(packet #{msg_count}, {age:.1f}s ago)"
        )
        live_class = "live-dot"
    elif mqtt_is_connecting(mqtt_state):
        live_text = "🔄 CONNECTING — waiting for MQTT packets from Wokwi"
        live_class = "live-dot"
    elif age is not None:
        live_text = (
            f"🔴 STALE DATA — website has not received a new MQTT packet "
            f"for {age:.0f}s (UI refresh is still running every 1s)"
        )
        live_class = "off-dot"
    else:
        live_text = "🟡 WAITING — no MQTT packet has reached the website yet"
        live_class = "off-dot"

    st.markdown(
        f'<div class="small-note"><span class="{live_class}">{live_text}</span></div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # GLOBAL KPIs
    # --------------------------------------------------------

    temperatures = [
        safe_float(server.get("temperature", 0))
        for server in servers
    ]

    average_temp = sum(temperatures) / len(temperatures)
    highest_temp = max(temperatures)

    hotspot_count = sum(
        temp >= WARNING_TEMP
        for temp in temperatures
    )

    critical_count = sum(
        temp >= CRITICAL_TEMP
        for temp in temperatures
    )

    cooling_count = sum(
        bool(server.get("cooling_active", False))
        for server in servers
    )

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric("Servers", len(servers))
    k2.metric("Average GPU", f"{average_temp:.1f} °C")
    k3.metric("Highest GPU", f"{highest_temp:.1f} °C")
    k4.metric("Hotspots", hotspot_count)
    k5.metric("Cooling Active", cooling_count)

    if critical_count:
        st.error(
            f"🚨 {critical_count} server(s) are at critical temperature."
        )

    # --------------------------------------------------------
    # SERVER CARDS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🖥️ DATA-CENTER NODE TELEMETRY (4 SERVERS)</div>',
        unsafe_allow_html=True,
    )

    columns = st.columns(4)

    for index in range(4):
        if index < len(servers):
            server = servers[index]
        else:
            server = {
                "server_id": index + 1,
                "temperature": 0,
                "predicted_temperature": 0,
                "cooling_active": False,
                "thermal_stress": 0,
            }

        sid = int(server.get("server_id", index + 1))
        temp = safe_float(server.get("temperature", 0))
        predicted = safe_float(
            server.get("predicted_temperature", temp)
        )
        previous = safe_float(
            server.get(
                "previous_temperature",
                temp - 0.5,
            )
        )

        cooling = bool(server.get("cooling_active", False))
        status = temperature_status(temp)
        badge_class = status_badge(temp)

        delta = temp - previous
        marker = max(0.0, min(100.0, temp / 60.0 * 100.0))

        stress = thermal_stress(
            temp,
            previous,
            cooling,
        )

        with columns[index]:
            card_html = f"""<div class=\"server-card\"><div class=\"server-header\"><div><div class=\"server-title\">🖥️ Server {sid:02d}</div><div class=\"server-rack\">Rack A-{sid:02d}</div></div><div><span class=\"badge {badge_class}\">◉ {status}</span><br><span class=\"badge badge-blue\" style=\"margin-top:6px;\">{'❄ COOLING ON' if cooling else '❄ COOLING OFF'}</span></div></div><div class=\"temp-panel\"><div class=\"temp-grid\"><div><div class=\"temp-label\">🌡 Current GPU</div><div class=\"temp-value\">{temp:.1f}°C</div><div class=\"delta\">~ {delta:+.1f}°C vs prev</div></div><div><div class=\"temp-label\">✣ AI Next Pred</div><div class=\"pred-value\">{predicted:.2f}°C</div><div class=\"delta\">Threshold {SAFE_TEMP:.0f}°C</div></div></div><div class=\"gauge\"><div class=\"gauge-marker\" style=\"left:{marker:.1f}%\"></div></div><div class=\"gauge-labels\"><span>20°C</span><span>35°C threshold</span><span>50°C</span></div></div><div class=\"info-grid\"><div class=\"info-box\"><div class=\"info-label\">Previous Temperature</div><div class=\"info-value\">{previous:.1f}°C</div></div><div class=\"info-box\"><div class=\"info-label\">Thermal Gauge</div><div class=\"info-value\">{temp:.1f}°C / 50°C</div></div><div class=\"info-box\"><div class=\"info-label\">Hardware</div><div class=\"info-value\">{status}</div></div><div class=\"info-box\"><div class=\"info-label\">Thermal Stress</div><div class=\"info-value\">{stress}</div></div></div><div class=\"formula-box\">AI: 0.8694 − (0.0371 × previous) + (1.0082 × current)</div></div>"""
            st.markdown(card_html, unsafe_allow_html=True)

    # --------------------------------------------------------
    # RAW MQTT
    # --------------------------------------------------------

    with st.expander("📨 Live MQTT Payload"):
        st.json(data)

    # --------------------------------------------------------
    # HARDWARE TABLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🔧 Real-Time Hardware Status</div>',
        unsafe_allow_html=True,
    )

    rows = []

    for server in servers:
        sid = server.get("server_id", 0)
        temp = safe_float(server.get("temperature", 0))
        predicted = safe_float(
            server.get("predicted_temperature", temp)
        )
        cooling = bool(server.get("cooling_active", False))

        rows.append(
            {
                "Server": f"Server {sid}",
                "Wokwi Temperature (°C)": round(temp, 2),
                "Predicted Temperature (°C)": round(predicted, 2),
                "Status": temperature_status(temp),
                "Hotspot": hotspot_status(server),
                "Cooling": "ON" if cooling else "OFF",
                "Protection": protection_status(temp),
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # SERVER DETAIL
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🔍 Server Detail Analysis</div>',
        unsafe_allow_html=True,
    )

    server_ids = [
        int(server.get("server_id", i + 1))
        for i, server in enumerate(servers)
    ]

    selected_server_id = st.selectbox(
        "Select Server",
        server_ids,
        key="server_selector",
    )

    selected_server = next(
        (
            server
            for server in servers
            if int(server.get("server_id", 0))
            == selected_server_id
        ),
        None,
    )

    if selected_server is not None:
        result, feature_df = predict_server(
            selected_server,
            models,
        )

        features = feature_df.iloc[0]

        current_temp = safe_float(
            selected_server.get("temperature", 0)
        )

        st.markdown(f"### Server {selected_server_id:02d}")

        d1, d2, d3, d4 = st.columns(4)

        d1.metric(
            "Current Wokwi Temperature",
            f"{current_temp:.2f} °C",
        )

        d2.metric(
            "AI Future GPU Temperature",
            f"{result['model1']:.2f} °C",
        )

        d3.metric(
            "AI Required Fan",
            f"{result['model2']:.1f}%",
        )

        d4.metric(
            "AI Hotspot Risk",
            f"{result['model3']:.1f}%",
        )

        # ----------------------------------------------------
        # ORIGINAL 11 PARAMETERS
        # ----------------------------------------------------

        st.markdown("### 📊 ML Input Parameters")

        st.caption(
            "Wokwi directly measures GPU/server temperature. "
            "The other parameters are estimated from the live "
            "temperature for this simulation."
        )

        p1, p2, p3, p4 = st.columns(4)

        p1.metric(
            "GPU Usage",
            f"{features['GPU_usage_percent']:.1f}%",
        )

        p2.metric(
            "GPU Power",
            f"{features['GPU_power_W']:.1f} W",
        )

        p3.metric(
            "CPU Usage",
            f"{features['CPU_usage_percent']:.1f}%",
        )

        p4.metric(
            "CPU Temperature",
            f"{features['CPU_temperature_C']:.1f} °C",
        )

        p5, p6, p7, p8 = st.columns(4)

        p5.metric(
            "GPU Temperature",
            f"{features['GPU_temperature_C']:.1f} °C",
        )

        p6.metric(
            "Ambient",
            f"{features['ambient_temperature_C']:.1f} °C",
        )

        p7.metric(
            "Inlet",
            f"{features['inlet_temperature_C']:.1f} °C",
        )

        p8.metric(
            "Outlet",
            f"{features['outlet_temperature_C']:.1f} °C",
        )

        p9, p10, p11 = st.columns(3)

        p9.metric(
            "Fan Speed",
            f"{features['fan_speed_percent']:.1f}%",
        )

        p10.metric(
            "Airflow",
            f"{features['airflow_m3_s']:.2f} m³/s",
        )

        p11.metric(
            "Workload",
            f"{features['workload_percent']:.1f}%",
        )

        # ----------------------------------------------------
        # MODEL RESULTS
        # ----------------------------------------------------

        st.markdown("### 🤖 AI Prediction Results")

        r1, r2, r3, r4, r5 = st.columns(5)

        r1.metric(
            "Model 1 — Future Temp",
            f"{result['model1']:.2f} °C",
        )

        r2.metric(
            "Model 2 — Required Fan",
            f"{result['model2']:.1f}%",
        )

        r3.metric(
            "Model 3 — Hotspot Risk",
            f"{result['model3']:.1f}%",
        )

        r4.metric(
            "Model 4 — Cooling",
            f"{result['model4']:.1f}%",
        )

        r5.metric(
            "Model 5 — Warning",
            "WARNING" if result["model5"] else "NORMAL",
        )

        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        st.markdown("### 🧠 AI Explanation")

        if current_temp >= CRITICAL_TEMP:
            st.error(
                "The Wokwi server is in the critical temperature range. "
                "The dashboard marks this as a critical hotspot and "
                "keeps the protection state active."
            )

        elif current_temp >= WARNING_TEMP:
            st.warning(
                "The Wokwi server is in the high-temperature range. "
                "The system identifies this as a hotspot/high thermal-stress condition."
            )

        elif result["model1"] >= SAFE_TEMP:
            st.info(
                "The AI model predicts that the server may approach "
                "the monitored hotspot threshold."
            )

        else:
            st.success(
                "The current Wokwi temperature and AI prediction are "
                "within the monitored range."
            )

    # --------------------------------------------------------
    # ANALYTICS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📈 Real-Time Analytics</div>',
        unsafe_allow_html=True,
    )

    chart_rows = []

    for server in servers:
        sid = int(server.get("server_id", 0))
        temp = safe_float(server.get("temperature", 0))

        result, _ = predict_server(
            server,
            models,
        )

        chart_rows.append(
            {
                "Server": f"Server {sid}",
                "Temperature": temp,
                "Prediction": safe_float(
                    server.get(
                        "predicted_temperature",
                        result["model1"],
                    )
                ),
                "Risk": result["model3"],
                "Cooling": result["model4"],
            }
        )

    chart_df = pd.DataFrame(chart_rows)

    # Temperature chart
    fig1 = go.Figure()

    fig1.add_trace(
        go.Bar(
            x=chart_df["Server"],
            y=chart_df["Temperature"],
            name="Current",
        )
    )

    fig1.add_trace(
        go.Scatter(
            x=chart_df["Server"],
            y=chart_df["Prediction"],
            name="AI Next Prediction",
            mode="lines+markers",
        )
    )

    fig1.add_hline(
        y=SAFE_TEMP,
        line_dash="dash",
        annotation_text="Safe 35°C",
    )

    fig1.add_hline(
        y=WARNING_TEMP,
        line_dash="dash",
        annotation_text="Warning 40°C",
    )

    fig1.add_hline(
        y=CRITICAL_TEMP,
        line_dash="dash",
        annotation_text="Critical 50°C",
    )

    fig1.update_layout(
        title="Live Wokwi Temperature vs AI Prediction",
        template="plotly_dark",
        height=430,
        xaxis_title="Server",
        yaxis_title="Temperature (°C)",
    )

    st.plotly_chart(
        fig1,
        use_container_width=True,
    )

    # Risk chart
    fig2 = go.Figure()

    fig2.add_trace(
        go.Bar(
            x=chart_df["Server"],
            y=chart_df["Risk"],
            name="Hotspot Risk",
        )
    )

    fig2.update_layout(
        title="AI Hotspot Risk",
        template="plotly_dark",
        height=400,
        xaxis_title="Server",
        yaxis_title="Risk (%)",
        yaxis_range=[0, 100],
    )

    st.plotly_chart(
        fig2,
        use_container_width=True,
    )

    # Cooling chart
    fig3 = go.Figure()

    fig3.add_trace(
        go.Bar(
            x=chart_df["Server"],
            y=chart_df["Cooling"],
            name="Cooling Effectiveness",
        )
    )

    fig3.update_layout(
        title="AI Cooling Effectiveness",
        template="plotly_dark",
        height=400,
        xaxis_title="Server",
        yaxis_title="Effectiveness (%)",
        yaxis_range=[0, 100],
    )

    st.plotly_chart(
        fig3,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📊 Model Performance</div>',
        unsafe_allow_html=True,
    )

    performance_rows = []

    for key, bundle in models.items():
        metrics = bundle["metrics"]

        row = {
            "Model": bundle["name"],
            "Target": bundle["target"],
            "Rows": metrics["rows"],
        }

        if bundle["kind"] == "classification":
            row["Accuracy"] = round(metrics["Accuracy"], 4)
            row["R²"] = None
            row["MAE"] = None
            row["RMSE"] = None
        else:
            row["Accuracy"] = None
            row["R²"] = round(metrics["R2"], 4)
            row["MAE"] = round(metrics["MAE"], 4)
            row["RMSE"] = round(metrics["RMSE"], 4)

        performance_rows.append(row)

    st.dataframe(
        pd.DataFrame(performance_rows),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# START
# ============================================================

if st.session_state.get("mqtt_enabled", False):
    # The fragment is rendered only during the live operation.
    # After DISCONNECT triggers st.rerun(), this block is skipped, so the
    # 1-second fragment refresh is no longer scheduled.
    realtime_dashboard()
else:
    st.markdown(
        """
<div class="disconnect-panel">
    <div class="disconnect-title">⏸️ SYSTEM DISCONNECTED</div>
    <div class="disconnect-text">
        MQTT listener stopped. Live Wokwi telemetry, ML processing, and
        automatic dashboard refresh are paused. Click CONNECT / RECONNECT
        to resume the complete live operation.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
