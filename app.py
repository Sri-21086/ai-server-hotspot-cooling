import json
import threading
import time

import pandas as pd
import streamlit as st
import paho.mqtt.client as mqtt
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Server Hotspot Cooling System",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# DARK THEME
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #0b0f14;
    color: #f5f5f5;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4 {
    color: #ffffff !important;
}

[data-testid="stSidebar"] {
    background-color: #11161d;
}

[data-testid="stMetric"] {
    background-color: #151b23;
    border: 1px solid #29313d;
    border-radius: 12px;
    padding: 12px;
}

.server-card {
    background-color: #151b23;
    border: 1px solid #29313d;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
}

.safe {
    color: #5ee58a;
    font-weight: 700;
}

.warning {
    color: #ffd166;
    font-weight: 700;
}

.critical {
    color: #ff5c5c;
    font-weight: 700;
}

.hotspot {
    color: #ff8c42;
    font-weight: 700;
}

.live {
    color: #5ee58a;
    font-weight: 700;
}

.small-text {
    color: #9aa4b2;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "ai-cooling/WOKWI-ESP32-001"

CSV1 = "server_cooling_model1_1000_rows.csv"
CSV2 = "server_cooling_model2_1000_rows.csv"
CSV3 = "server_cooling_model3_1000_rows.csv"
CSV4 = "server_cooling_model4_1000_rows.csv"
CSV5 = "server_cooling_model5_1000_rows.csv"

FEATURES = [
    "GPU_usage_percent",
    "GPU_power_W",
    "CPU_usage_percent",
    "CPU_temperature_C",
    "GPU_temperature_C",
    "ambient_temperature_C",
    "inlet_temperature_C",
    "outlet_temperature_C",
    "fan_speed_percent",
    "airflow_m3_s",
    "workload_percent"
]

SAFE_TEMP = 35.0
WARNING_TEMP = 40.0
CRITICAL_TEMP = 50.0


# ============================================================
# MQTT STATE
# ============================================================

class MQTTState:

    def __init__(self):
        self.lock = threading.Lock()

        self.connected = False
        self.latest_data = None
        self.last_update = None

        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

    def on_connect(self, client, userdata, flags, reason_code, properties):

        print("MQTT CONNECT RESULT:", reason_code)

        if reason_code == 0:

            with self.lock:
                self.connected = True

            print("MQTT connected successfully")

            client.subscribe(
                MQTT_TOPIC,
                qos=0
            )

            print("Subscribed to:", MQTT_TOPIC)

        else:

            with self.lock:
                self.connected = False

            print("MQTT connection failed:", reason_code)

    def on_message(self, client, userdata, msg):

        try:

            payload = msg.payload.decode("utf-8")

            data = json.loads(payload)

            with self.lock:

                self.latest_data = data
                self.last_update = time.time()

            print("NEW MQTT MESSAGE:")
            print(payload)

        except Exception as e:

            print("MQTT message processing error:", e)

    def on_disconnect(
        self,
        client,
        userdata,
        disconnect_flags,
        reason_code,
        properties
    ):

        with self.lock:
            self.connected = False

        print("MQTT disconnected:", reason_code)

    def connect(self):

        try:

            if self.client.is_connected():

                with self.lock:
                    self.connected = True

                return True

            self.client.connect(
                MQTT_BROKER,
                MQTT_PORT,
                keepalive=60
            )

            # IMPORTANT:
            # MQTT keeps listening in the background.
            self.client.loop_start()

            return True

        except Exception as e:

            print("MQTT connection error:", e)

            with self.lock:
                self.connected = False

            return False

    def disconnect(self):

        try:

            self.client.loop_stop()
            self.client.disconnect()

        except Exception:
            pass

        with self.lock:
            self.connected = False

    def get_data(self):

        with self.lock:
            return self.latest_data

    def is_connected(self):

        with self.lock:
            return self.connected

    def get_last_update(self):

        with self.lock:
            return self.last_update


@st.cache_resource
def get_mqtt_state():

    return MQTTState()


mqtt_state = get_mqtt_state()


# ============================================================
# SESSION STATE
# ============================================================

if "mqtt_enabled" not in st.session_state:
    st.session_state.mqtt_enabled = False

if "selected_server" not in st.session_state:
    st.session_state.selected_server = 1


# ============================================================
# LOAD DATASETS
# ============================================================

@st.cache_data
def load_datasets():

    df1 = pd.read_csv(CSV1)
    df2 = pd.read_csv(CSV2)
    df3 = pd.read_csv(CSV3)
    df4 = pd.read_csv(CSV4)
    df5 = pd.read_csv(CSV5)

    return df1, df2, df3, df4, df5


# ============================================================
# TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models():

    df1, df2, df3, df4, df5 = load_datasets()

    models = {}

    # -------------------------
    # MODEL 1
    # Future GPU Temperature
    # -------------------------

    X1 = df1[FEATURES]
    y1 = df1["future_GPU_temperature_C"]

    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1,
        y1,
        test_size=0.2,
        random_state=42
    )

    model1 = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    model1.fit(X1_train, y1_train)

    pred1 = model1.predict(X1_test)

    models["model1"] = model1

    models["model1_metrics"] = {
        "R2": r2_score(y1_test, pred1),
        "MAE": mean_absolute_error(y1_test, pred1),
        "RMSE": mean_squared_error(
            y1_test,
            pred1
        ) ** 0.5
    }

    # -------------------------
    # MODEL 2
    # Required Fan Speed
    # -------------------------

    X2 = df2[FEATURES]
    y2 = df2["required_fan_speed_percent"]

    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2,
        y2,
        test_size=0.2,
        random_state=42
    )

    model2 = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    model2.fit(X2_train, y2_train)

    pred2 = model2.predict(X2_test)

    models["model2"] = model2

    models["model2_metrics"] = {
        "R2": r2_score(y2_test, pred2),
        "MAE": mean_absolute_error(y2_test, pred2),
        "RMSE": mean_squared_error(
            y2_test,
            pred2
        ) ** 0.5
    }

    # -------------------------
    # MODEL 3
    # Hotspot Risk
    # -------------------------

    X3 = df3[FEATURES]
    y3 = df3["hotspot_risk_percent"]

    X3_train, X3_test, y3_train, y3_test = train_test_split(
        X3,
        y3,
        test_size=0.2,
        random_state=42
    )

    model3 = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    model3.fit(X3_train, y3_train)

    pred3 = model3.predict(X3_test)

    models["model3"] = model3

    models["model3_metrics"] = {
        "R2": r2_score(y3_test, pred3),
        "MAE": mean_absolute_error(y3_test, pred3),
        "RMSE": mean_squared_error(
            y3_test,
            pred3
        ) ** 0.5
    }

    # -------------------------
    # MODEL 4
    # Cooling Effectiveness
    # -------------------------

    X4 = df4[FEATURES]
    y4 = df4["cooling_effectiveness_percent"]

    X4_train, X4_test, y4_train, y4_test = train_test_split(
        X4,
        y4,
        test_size=0.2,
        random_state=42
    )

    model4 = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    model4.fit(X4_train, y4_train)

    pred4 = model4.predict(X4_test)

    models["model4"] = model4

    models["model4_metrics"] = {
        "R2": r2_score(y4_test, pred4),
        "MAE": mean_absolute_error(y4_test, pred4),
        "RMSE": mean_squared_error(
            y4_test,
            pred4
        ) ** 0.5
    }

    # -------------------------
    # MODEL 5
    # Overheating Warning
    # -------------------------

    X5 = df5[FEATURES]
    y5 = df5["overheating_warning"].astype(int)

    model5 = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    model5.fit(X5, y5)

    models["model5"] = model5

    return models


models = train_models()


# ============================================================
# BUILD ML INPUT FEATURES FROM LIVE TEMPERATURE
# ============================================================

def build_features(server):

    temp = float(
        server.get("temperature", 22.0)
    )

    predicted = float(
        server.get(
            "predicted_temperature",
            temp
        )
    )

    cooling = bool(
        server.get(
            "cooling_active",
            False
        )
    )

    thermal_stress = float(
        server.get(
            "thermal_stress",
            0.0
        )
    )

    # These are estimated/simulated inputs.
    gpu_usage = min(
        100.0,
        max(
            10.0,
            35.0 + (temp - 25.0) * 4.0
        )
    )

    gpu_power = min(
        450.0,
        max(
            80.0,
            120.0 + gpu_usage * 2.2
        )
    )

    cpu_usage = min(
        100.0,
        max(
            15.0,
            gpu_usage * 0.72
        )
    )

    cpu_temp = min(
        90.0,
        max(
            30.0,
            35.0 + (temp - 25.0) * 0.65
        )
    )

    ambient = 25.0

    inlet = max(
        18.0,
        temp - 8.0
    )

    outlet = temp + 5.0

    fan_speed = min(
        100.0,
        max(
            25.0,
            40.0 + (temp - 30.0) * 3.0
        )
    )

    if cooling:
        fan_speed = min(
            100.0,
            fan_speed + 15.0
        )

    airflow = min(
        3.0,
        max(
            0.4,
            0.8 + fan_speed / 100.0 * 1.5
        )
    )

    workload = min(
        100.0,
        max(
            10.0,
            gpu_usage * 0.92
        )
    )

    return pd.DataFrame([{
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
        "workload_percent": workload
    }])


# ============================================================
# PREDICT ALL MODELS
# ============================================================

def predict_server(server):

    features = build_features(server)

    future_temp = float(
        models["model1"].predict(features)[0]
    )

    required_fan = float(
        models["model2"].predict(features)[0]
    )

    hotspot_risk = float(
        models["model3"].predict(features)[0]
    )

    cooling_effectiveness = float(
        models["model4"].predict(features)[0]
    )

    warning_prediction = int(
        models["model5"].predict(features)[0]
    )

    temperature = float(
        server.get("temperature", 0)
    )

    # --------------------------------------------------------
    # Safety override
    # Ensures actual live temperature is respected.
    # --------------------------------------------------------

    if temperature >= CRITICAL_TEMP:

        hotspot_risk = max(
            hotspot_risk,
            100.0
        )

        warning_prediction = 1

    elif temperature >= WARNING_TEMP:

        hotspot_risk = max(
            hotspot_risk,
            80.0
        )

        warning_prediction = 1

    elif temperature >= SAFE_TEMP:

        hotspot_risk = max(
            hotspot_risk,
            35.0
        )

    hotspot_risk = max(
        0.0,
        min(
            100.0,
            hotspot_risk
        )
    )

    cooling_effectiveness = max(
        0.0,
        min(
            100.0,
            cooling_effectiveness
        )
    )

    required_fan = max(
        0.0,
        min(
            100.0,
            required_fan
        )
    )

    return {
        "future_temp": future_temp,
        "required_fan": required_fan,
        "hotspot_risk": hotspot_risk,
        "cooling_effectiveness": cooling_effectiveness,
        "warning": warning_prediction
    }


# ============================================================
# STATUS FUNCTIONS
# ============================================================

def get_temperature_status(temp):

    if temp >= CRITICAL_TEMP:
        return "CRITICAL"

    if temp >= WARNING_TEMP:
        return "HIGH TEMPERATURE"

    if temp >= SAFE_TEMP:
        return "WARM - MONITORING"

    return "SAFE"


def get_status_class(temp):

    if temp >= CRITICAL_TEMP:
        return "critical"

    if temp >= WARNING_TEMP:
        return "hotspot"

    if temp >= SAFE_TEMP:
        return "warning"

    return "safe"


def get_hotspot_status(server, ml_result):

    temp = float(
        server.get("temperature", 0)
    )

    predicted = float(
        server.get(
            "predicted_temperature",
            temp
        )
    )

    cooling = bool(
        server.get(
            "cooling_active",
            False
        )
    )

    # Actual temperature has priority.

    if temp >= CRITICAL_TEMP:
        return "CRITICAL HOTSPOT"

    if temp >= WARNING_TEMP:
        return "HOTSPOT / HIGH THERMAL STRESS"

    if predicted >= SAFE_TEMP:
        return "HOTSPOT PREDICTED"

    if cooling:
        return "COOLING ACTIVE"

    return "NO HOTSPOT"


def get_protection_status(temp):

    if temp >= CRITICAL_TEMP:
        return "CRITICAL PROTECTION"

    if temp >= WARNING_TEMP:
        return "ACTIVE"

    return "NORMAL"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ System Control")

    st.markdown(
        "### MQTT Device"
    )

    st.code(
        "WOKWI-ESP32-001",
        language=None
    )

    st.markdown(
        f"**Broker:** `{MQTT_BROKER}`"
    )

    st.markdown(
        f"**Topic:** `{MQTT_TOPIC}`"
    )

    st.divider()

    if not st.session_state.mqtt_enabled:

        if st.button(
            "🔌 CONNECT",
            use_container_width=True
        ):

            success = mqtt_state.connect()

            if success:

                st.session_state.mqtt_enabled = True

                st.rerun()

            else:

                st.error(
                    "MQTT connection failed."
                )

    else:

        if st.button(
            "🔴 DISCONNECT",
            use_container_width=True
        ):

            mqtt_state.disconnect()

            st.session_state.mqtt_enabled = False

            st.rerun()

    st.divider()

    if mqtt_state.is_connected():

        st.success(
            "🟢 MQTT CONNECTED"
        )

    else:

        st.warning(
            "🟡 MQTT NOT CONNECTED"
        )

    st.divider()

    st.markdown("### Simulation")

    st.write(
        "Servers simulated: **4**"
    )

    st.write(
        "Update interval: **3 seconds**"
    )

    st.divider()

    st.caption(
        "AI Server Hotspot Cooling System"
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🌡️ AI Server Hotspot Cooling System"
)

st.markdown(
    "Real-time server thermal monitoring, hotspot prediction and AI-based cooling analysis."
)


# ============================================================
# REALTIME DASHBOARD
# ============================================================

@st.fragment(run_every=3)
def realtime_dashboard():

    data = mqtt_state.get_data()

    # --------------------------------------------------------
    # NO DATA YET
    # --------------------------------------------------------

    if data is None:

        st.info(
            "Waiting for live data from Wokwi..."
        )

        st.caption(
            "Connect the MQTT dashboard and keep the Wokwi simulation running."
        )

        return

    servers = data.get(
        "servers",
        []
    )

    if not servers:

        st.warning(
            "MQTT message received, but no server data was found."
        )

        return

    # --------------------------------------------------------
    # LIVE STATUS
    # --------------------------------------------------------

    last_update = mqtt_state.get_last_update()

    if last_update is not None:

        elapsed = max(
            0,
            time.time() - last_update
        )

        if elapsed <= 6:

            st.markdown(
                '<span class="live">🟢 LIVE — MQTT data updating</span>',
                unsafe_allow_html=True
            )

        else:

            st.warning(
                "MQTT connected, but no new message received recently."
            )

    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

    temperatures = [
        float(
            s.get("temperature", 0)
        )
        for s in servers
    ]

    cooling_count = sum(
        bool(
            s.get(
                "cooling_active",
                False
            )
        )
        for s in servers
    )

    hotspot_count = sum(
        float(
            s.get(
                "temperature",
                0
            )
        ) >= WARNING_TEMP
        for s in servers
    )

    critical_count = sum(
        float(
            s.get(
                "temperature",
                0
            )
        ) >= CRITICAL_TEMP
        for s in servers
    )

    average_temp = (
        sum(temperatures) /
        len(temperatures)
        if temperatures
        else 0
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Servers",
        len(servers)
    )

    c2.metric(
        "Average Temperature",
        f"{average_temp:.1f} °C"
    )

    c3.metric(
        "Highest Temperature",
        f"{max(temperatures):.1f} °C"
    )

    c4.metric(
        "Hotspots",
        hotspot_count
    )

    c5.metric(
        "Cooling Active",
        cooling_count
    )

    if critical_count > 0:

        st.error(
            f"🚨 {critical_count} server(s) at CRITICAL temperature."
        )

    # --------------------------------------------------------
    # SERVER CARDS
    # --------------------------------------------------------

    st.markdown("## 🖥️ Live Server Monitoring")

    cols = st.columns(2)

    processed_servers = []

    for index, server in enumerate(servers):

        server_id = int(
            server.get(
                "server_id",
                index + 1
            )
        )

        temperature = float(
            server.get(
                "temperature",
                0
            )
        )

        predicted_temperature = float(
            server.get(
                "predicted_temperature",
                temperature
            )
        )

        cooling_active = bool(
            server.get(
                "cooling_active",
                False
            )
        )

        thermal_stress = float(
            server.get(
                "thermal_stress",
                0
            )
        )

        status = get_temperature_status(
            temperature
        )

        hotspot = get_hotspot_status(
            server,
            None
        )

        protection = get_protection_status(
            temperature
        )

        ml_result = predict_server(
            server
        )

        processed_servers.append({
            "server": server,
            "ml": ml_result
        })

        with cols[index % 2]:

            st.markdown(
                '<div class="server-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                f"### 🖥️ Server {server_id}"
            )

            st.markdown(
                f'<span class="{get_status_class(temperature)}">'
                f'{status}'
                f'</span>',
                unsafe_allow_html=True
            )

            st.markdown("")

            a, b = st.columns(2)

            a.metric(
                "Current",
                f"{temperature:.2f} °C"
            )

            b.metric(
                "Predicted",
                f"{predicted_temperature:.2f} °C"
            )

            c, d = st.columns(2)

            c.metric(
                "Cooling",
                "ON" if cooling_active else "OFF"
            )

            d.metric(
                "Thermal Stress",
                f"{thermal_stress:.1f}"
            )

            if temperature >= CRITICAL_TEMP:

                st.error(
                    "🚨 CRITICAL HOTSPOT"
                )

            elif temperature >= WARNING_TEMP:

                st.warning(
                    "🔥 HOTSPOT / HIGH THERMAL STRESS"
                )

            elif predicted_temperature >= SAFE_TEMP:

                st.info(
                    "🔮 HOTSPOT PREDICTED"
                )

            else:

                st.success(
                    "✓ NO HOTSPOT"
                )

            st.write(
                f"**Protection:** {protection}"
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

    # --------------------------------------------------------
    # HARDWARE STATUS TABLE
    # --------------------------------------------------------

    st.markdown("## 🔧 Real-Time Hardware Status")

    hardware_rows = []

    for server in servers:

        sid = int(
            server.get(
                "server_id",
                0
            )
        )

        temp = float(
            server.get(
                "temperature",
                0
            )
        )

        predicted = float(
            server.get(
                "predicted_temperature",
                temp
            )
        )

        cooling = bool(
            server.get(
                "cooling_active",
                False
            )
        )

        hardware_rows.append({
            "Server": f"Server {sid}",
            "Temperature (°C)": round(
                temp,
                2
            ),
            "Predicted (°C)": round(
                predicted,
                2
            ),
            "Status": get_temperature_status(
                temp
            ),
            "Hotspot": get_hotspot_status(
                server,
                None
            ),
            "Cooling": "ON" if cooling else "OFF",
            "Protection": get_protection_status(
                temp
            )
        })

    st.dataframe(
        pd.DataFrame(hardware_rows),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # SERVER DETAIL
    # --------------------------------------------------------

    st.markdown("## 🔍 Server Detail Analysis")

    server_numbers = [
        int(
            s.get(
                "server_id",
                i + 1
            )
        )
        for i, s in enumerate(servers)
    ]

    selected = st.selectbox(
        "Select Server",
        server_numbers,
        index=0
    )

    selected_server = None

    for server in servers:

        if int(
            server.get(
                "server_id",
                0
            )
        ) == selected:

            selected_server = server
            break

    if selected_server is not None:

        ml = predict_server(
            selected_server
        )

        features = build_features(
            selected_server
        ).iloc[0]

        st.markdown(
            f"### Server {selected}"
        )

        d1, d2, d3, d4 = st.columns(4)

        d1.metric(
            "Current Temperature",
            f'{selected_server.get("temperature", 0):.2f} °C'
        )

        d2.metric(
            "Future GPU Temperature",
            f'{ml["future_temp"]:.2f} °C'
        )

        d3.metric(
            "Required Fan Speed",
            f'{ml["required_fan"]:.1f}%'
        )

        d4.metric(
            "Hotspot Risk",
            f'{ml["hotspot_risk"]:.1f}%'
        )

        # ----------------------------------------------------
        # EXTRA ML INPUTS
        # ----------------------------------------------------

        st.markdown(
            "### 📊 Estimated ML Input Parameters"
        )

        f1, f2, f3, f4 = st.columns(4)

        f1.metric(
            "GPU Usage",
            f'{features["GPU_usage_percent"]:.1f}%'
        )

        f2.metric(
            "GPU Power",
            f'{features["GPU_power_W"]:.1f} W'
        )

        f3.metric(
            "CPU Usage",
            f'{features["CPU_usage_percent"]:.1f}%'
        )

        f4.metric(
            "CPU Temperature",
            f'{features["CPU_temperature_C"]:.1f} °C'
        )

        f5, f6, f7, f8 = st.columns(4)

        f5.metric(
            "Ambient",
            f'{features["ambient_temperature_C"]:.1f} °C'
        )

        f6.metric(
            "Inlet",
            f'{features["inlet_temperature_C"]:.1f} °C'
        )

        f7.metric(
            "Outlet",
            f'{features["outlet_temperature_C"]:.1f} °C'
        )

        f8.metric(
            "Airflow",
            f'{features["airflow_m3_s"]:.2f} m³/s'
        )

        f9, f10 = st.columns(2)

        f9.metric(
            "Workload",
            f'{features["workload_percent"]:.1f}%'
        )

        f10.metric(
            "Fan Speed",
            f'{features["fan_speed_percent"]:.1f}%'
        )

        # ----------------------------------------------------
        # AI RESULTS
        # ----------------------------------------------------

        st.markdown(
            "### 🤖 AI Prediction Results"
        )

        r1, r2, r3, r4, r5 = st.columns(5)

        r1.metric(
            "Model 1",
            f'{ml["future_temp"]:.2f} °C'
        )

        r2.metric(
            "Model 2",
            f'{ml["required_fan"]:.1f}%'
        )

        r3.metric(
            "Model 3",
            f'{ml["hotspot_risk"]:.1f}%'
        )

        r4.metric(
            "Model 4",
            f'{ml["cooling_effectiveness"]:.1f}%'
        )

        r5.metric(
            "Model 5",
            "WARNING" if ml["warning"] else "NORMAL"
        )

        # ----------------------------------------------------
        # AI EXPLANATION
        # ----------------------------------------------------

        st.markdown(
            "### 🧠 AI Explanation"
        )

        temp = float(
            selected_server.get(
                "temperature",
                0
            )
        )

        if temp >= CRITICAL_TEMP:

            st.error(
                "The server is currently in the critical temperature range. "
                "Cooling protection should remain active."
            )

        elif temp >= WARNING_TEMP:

            st.warning(
                "The server is in a high-temperature range. "
                "The system identifies this as a hotspot/high thermal-stress condition."
            )

        elif ml["future_temp"] >= SAFE_TEMP:

            st.info(
                "The AI prediction indicates that the server may reach "
                "the hotspot threshold, so cooling action is recommended."
            )

        else:

            st.success(
                "The current thermal condition is within the monitored safe range."
            )

        # ----------------------------------------------------
        # DATA NOTE
        # ----------------------------------------------------

        st.caption(
            "Note: Wokwi provides the simulated hardware temperature. "
            "Other ML input parameters are estimated/simulated from the live temperature "
            "for demonstration of the AI models."
        )

    # ========================================================
    # GRAPHS AT END
    # ========================================================

    st.markdown("## 📈 Real-Time Analytics")

    chart_servers = []

    for server in servers:

        sid = int(
            server.get(
                "server_id",
                0
            )
        )

        temp = float(
            server.get(
                "temperature",
                0
            )
        )

        ml = predict_server(
            server
        )

        chart_servers.append({
            "Server": f"Server {sid}",
            "Temperature": temp,
            "Risk": ml["hotspot_risk"],
            "Cooling": ml["cooling_effectiveness"]
        })

    chart_df = pd.DataFrame(
        chart_servers
    )

    # --------------------------------------------------------
    # TEMPERATURE CHART
    # --------------------------------------------------------

    fig_temp = go.Figure()

    fig_temp.add_trace(
        go.Bar(
            x=chart_df["Server"],
            y=chart_df["Temperature"],
            name="Temperature"
        )
    )

    fig_temp.add_hline(
        y=SAFE_TEMP,
        line_dash="dash",
        annotation_text="Safe threshold"
    )

    fig_temp.add_hline(
        y=WARNING_TEMP,
        line_dash="dash",
        annotation_text="Warning threshold"
    )

    fig_temp.add_hline(
        y=CRITICAL_TEMP,
        line_dash="dash",
        annotation_text="Critical threshold"
    )

    fig_temp.update_layout(
        title="Current Server Temperatures",
        template="plotly_dark",
        height=420,
        xaxis_title="Server",
        yaxis_title="Temperature (°C)"
    )

    st.plotly_chart(
        fig_temp,
        use_container_width=True
    )

    # --------------------------------------------------------
    # HOTSPOT RISK
    # --------------------------------------------------------

    fig_risk = go.Figure()

    fig_risk.add_trace(
        go.Bar(
            x=chart_df["Server"],
            y=chart_df["Risk"],
            name="Hotspot Risk"
        )
    )

    fig_risk.update_layout(
        title="AI Hotspot Risk",
        template="plotly_dark",
        height=420,
        xaxis_title="Server",
        yaxis_title="Risk (%)",
        yaxis_range=[0, 100]
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True
    )

    # --------------------------------------------------------
    # COOLING EFFECTIVENESS
    # --------------------------------------------------------

    fig_cooling = go.Figure()

    fig_cooling.add_trace(
        go.Bar(
            x=chart_df["Server"],
            y=chart_df["Cooling"],
            name="Cooling Effectiveness"
        )
    )

    fig_cooling.update_layout(
        title="AI Cooling Effectiveness",
        template="plotly_dark",
        height=420,
        xaxis_title="Server",
        yaxis_title="Effectiveness (%)",
        yaxis_range=[0, 100]
    )

    st.plotly_chart(
        fig_cooling,
        use_container_width=True
    )

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.markdown("## 📊 Model Performance")

    performance = pd.DataFrame([
        {
            "Model": "Model 1 - Future GPU Temperature",
            "R²": models["model1_metrics"]["R2"],
            "MAE": models["model1_metrics"]["MAE"],
            "RMSE": models["model1_metrics"]["RMSE"]
        },
        {
            "Model": "Model 2 - Required Fan Speed",
            "R²": models["model2_metrics"]["R2"],
            "MAE": models["model2_metrics"]["MAE"],
            "RMSE": models["model2_metrics"]["RMSE"]
        },
        {
            "Model": "Model 3 - Hotspot Risk",
            "R²": models["model3_metrics"]["R2"],
            "MAE": models["model3_metrics"]["MAE"],
            "RMSE": models["model3_metrics"]["RMSE"]
        },
        {
            "Model": "Model 4 - Cooling Effectiveness",
            "R²": models["model4_metrics"]["R2"],
            "MAE": models["model4_metrics"]["MAE"],
            "RMSE": models["model4_metrics"]["RMSE"]
        }
    ])

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# START REALTIME DASHBOARD
# ============================================================

realtime_dashboard()
