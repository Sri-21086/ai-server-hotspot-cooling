import json
import os
import threading
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import paho.mqtt.client as mqtt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="AI Server Hotspot Cooling",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CONFIG
# =========================================================

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "ai-cooling/WOKWI-ESP32-001"

SAFE_TEMP = 35.0
WARNING_TEMP = 40.0
CRITICAL_TEMP = 50.0

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

MODEL_FILES = {
    "Model 1": (
        "server_cooling_model1_1000_rows.csv",
        "future_GPU_temperature_C"
    ),
    "Model 2": (
        "server_cooling_model2_1000_rows.csv",
        "required_fan_speed_percent"
    ),
    "Model 3": (
        "server_cooling_model3_1000_rows.csv",
        "hotspot_risk_percent"
    ),
    "Model 4": (
        "server_cooling_model4_1000_rows.csv",
        "cooling_effectiveness_percent"
    ),
    "Model 5": (
        "server_cooling_model5_1000_rows.csv",
        "overheating_warning"
    )
}


# =========================================================
# DARK THEME
# =========================================================

st.markdown("""
<style>

.stApp {
    background: #0b1020;
    color: #f1f5f9;
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 4px;
    border: none;
    box-shadow: none;
}

.subtitle {
    color: #94a3b8;
    font-size: 15px;
    margin-bottom: 24px;
}

.section-title {
    font-size: 23px;
    font-weight: 750;
    color: #f8fafc;
    margin-top: 25px;
    margin-bottom: 14px;
}

.card {
    background: #111827;
    border: 1px solid #263247;
    border-radius: 15px;
    padding: 18px;
}

[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #263247;
    padding: 15px;
    border-radius: 14px;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}

[data-testid="stMetricValue"] {
    color: #f8fafc !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #263247;
    border-radius: 12px;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    font-weight: 700;
}

div[data-testid="stSidebar"] {
    background: #080d19;
    border-right: 1px solid #202b3d;
}

div[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

.warning-box {
    background: #3a1f08;
    border: 1px solid #9a5b13;
    padding: 14px;
    border-radius: 12px;
}

.critical-box {
    background: #3b1015;
    border: 1px solid #b4232f;
    padding: 14px;
    border-radius: 12px;
}

.safe-box {
    background: #09251b;
    border: 1px solid #16734b;
    padding: 14px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "mqtt_enabled" not in st.session_state:
    st.session_state.mqtt_enabled = False


# =========================================================
# MQTT SYSTEM
# =========================================================

@st.cache_resource
def create_mqtt_system():

    class MQTTSystem:

        def __init__(self):
            self.data = None
            self.connected = False
            self.lock = threading.Lock()

            self.client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2
            )

            self.client.on_connect = self.on_connect
            self.client.on_message = self.on_message
            self.client.on_disconnect = self.on_disconnect

        def start(self):

            try:
                if not self.client.is_connected():

                    self.client.connect(
                        MQTT_BROKER,
                        MQTT_PORT,
                        60
                    )

                    self.client.loop_start()

                return True

            except Exception:
                self.connected = False
                return False

        def stop(self):

            try:
                if self.client.is_connected():
                    self.client.unsubscribe(MQTT_TOPIC)
                    self.client.disconnect()

                self.connected = False

            except Exception:
                pass

        def on_connect(
            self,
            client,
            userdata,
            flags,
            reason_code,
            properties
        ):

            if reason_code == 0:

                self.connected = True

                client.subscribe(
                    MQTT_TOPIC
                )

        def on_disconnect(
            self,
            client,
            userdata,
            disconnect_flags,
            reason_code,
            properties=None
        ):

            self.connected = False

        def on_message(
            self,
            client,
            userdata,
            msg
        ):

            try:

                payload = json.loads(
                    msg.payload.decode()
                )

                with self.lock:
                    self.data = payload

            except Exception:
                pass

        def get_data(self):

            with self.lock:

                if self.data is None:
                    return None

                return json.loads(
                    json.dumps(self.data)
                )

    return MQTTSystem()


mqtt_system = create_mqtt_system()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## ⚙️ System Control")

    device_id = st.text_input(
        "Device ID",
        value="WOKWI-ESP32-001"
    )

    server_count = st.number_input(
        "Number of Servers",
        min_value=1,
        max_value=20,
        value=4,
        step=1
    )

    st.divider()

    if not st.session_state.mqtt_enabled:

        if st.button(
            "🔌 CONNECT",
            use_container_width=True
        ):

            success = mqtt_system.start()

            if success:
                st.session_state.mqtt_enabled = True
                st.rerun()

    else:

        if st.button(
            "⛔ DISCONNECT",
            use_container_width=True
        ):

            mqtt_system.stop()
            st.session_state.mqtt_enabled = False
            st.rerun()

    st.divider()

    if st.session_state.mqtt_enabled:

        if mqtt_system.connected:
            st.success("🟢 MQTT CONNECTED")
        else:
            st.warning("🟡 CONNECTING...")

    else:

        st.info("⚪ SYSTEM DISCONNECTED")

    st.caption(
        f"Broker: {MQTT_BROKER}"
    )

    st.caption(
        f"Topic: {MQTT_TOPIC}"
    )

    st.divider()

    st.markdown("### 🌡️ Temperature Limits")

    st.write("🟢 SAFE — below 35°C")
    st.write("🟡 WARM — 35°C to 39.9°C")
    st.write("🟠 HIGH — 40°C to 49.9°C")
    st.write("🔴 CRITICAL — 50°C or above")


# =========================================================
# LOAD ML MODELS
# =========================================================

@st.cache_resource
def train_models():

    models = {}
    metrics = {}

    for model_name, (filename, target) in MODEL_FILES.items():

        if not os.path.exists(filename):
            continue

        df = pd.read_csv(filename)

        X = df[FEATURES]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        if target == "overheating_warning":

            model = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                random_state=42
            )

            model.fit(
                X_train,
                y_train
            )

            pred = model.predict(
                X_test
            )

            metrics[model_name] = {
                "accuracy":
                    accuracy_score(
                        y_test,
                        pred
                    )
            }

        else:

            model = RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                random_state=42
            )

            model.fit(
                X_train,
                y_train
            )

            pred = model.predict(
                X_test
            )

            metrics[model_name] = {
                "r2":
                    r2_score(
                        y_test,
                        pred
                    ),

                "mae":
                    mean_absolute_error(
                        y_test,
                        pred
                    )
            }

        models[model_name] = model

    return models, metrics


models, model_metrics = train_models()


# =========================================================
# SERVER FEATURE ESTIMATION
# =========================================================

def build_features(server):

    temp = float(
        server.get(
            "temperature",
            25
        )
    )

    cooling = bool(
        server.get(
            "cooling_active",
            False
        )
    )

    gpu_usage = max(
        10,
        min(
            100,
            15 + (temp - 20) * 2.2
        )
    )

    gpu_power = max(
        50,
        min(
            420,
            90 + gpu_usage * 2.5
        )
    )

    cpu_usage = max(
        10,
        min(
            100,
            25 + (temp - 20) * 1.5
        )
    )

    cpu_temp = max(
        28,
        min(
            75,
            30 + cpu_usage * 0.25
        )
    )

    ambient = 24.0

    inlet = (
        ambient +
        max(
            0,
            temp - 35
        ) * 0.08
    )

    outlet = (
        inlet +
        max(
            0,
            temp - 30
        ) * 0.18
    )

    fan_speed = 75 if cooling else 35

    airflow = (
        1.0 +
        fan_speed / 100
    )

    workload = (
        gpu_usage +
        cpu_usage
    ) / 2

    return {
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
    }


# =========================================================
# ML PREDICTIONS
# =========================================================

def predict_server(server):

    features = build_features(
        server
    )

    X = pd.DataFrame(
        [features],
        columns=FEATURES
    )

    results = {}

    for model_name, model in models.items():

        try:

            value = model.predict(X)[0]

            if model_name == "Model 1":

                results["future_temperature"] = float(
                    value
                )

            elif model_name == "Model 2":

                results["required_fan"] = float(
                    value
                )

            elif model_name == "Model 3":

                results["hotspot_risk"] = float(
                    value
                )

            elif model_name == "Model 4":

                results["cooling_effectiveness"] = float(
                    value
                )

            elif model_name == "Model 5":

                results["overheating_warning"] = int(
                    value
                )

        except Exception:

            pass

    return features, results


# =========================================================
# HARDWARE STATUS
# =========================================================

def get_status(temp):

    if temp >= CRITICAL_TEMP:
        return "CRITICAL"

    if temp >= WARNING_TEMP:
        return "HIGH TEMPERATURE"

    if temp >= SAFE_TEMP:
        return "WARM - MONITORING"

    return "SAFE"


def get_hotspot_status(
    temp,
    predicted_temp,
    cooling
):

    # Same protection concept as the Wokwi logic.
    # 40°C+ is never shown as "NO HOTSPOT".

    if temp >= CRITICAL_TEMP:

        return "CRITICAL HOTSPOT"

    if temp >= WARNING_TEMP:

        return "HOTSPOT / HIGH THERMAL STRESS"

    if predicted_temp >= SAFE_TEMP:

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


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">❄️ AI SERVER HOTSPOT COOLING SYSTEM</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Real-time thermal monitoring • MQTT • Wokwi ESP32 • AI prediction'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# REAL-TIME DASHBOARD
# =========================================================

@st.fragment(run_every=3)
def realtime_dashboard():

    if not st.session_state.mqtt_enabled:

        st.markdown(
            """
            <div class="warning-box">
            🔌 <b>System is disconnected.</b><br>
            Click CONNECT from the sidebar to start receiving Wokwi readings.
            </div>
            """,
            unsafe_allow_html=True
        )

        return

    live_data = mqtt_system.get_data()

    if live_data is None:

        st.warning(
            "Waiting for Wokwi data..."
        )

        return

    servers = live_data.get(
        "servers",
        []
    )

    servers = servers[
        :int(server_count)
    ]

    if not servers:

        st.warning(
            "No server readings received."
        )

        return


    # =====================================================
    # PROCESS DATA
    # =====================================================

    rows = []

    for server in servers:

        server_id = int(
            server.get(
                "server_id",
                len(rows) + 1
            )
        )

        temperature = float(
            server.get(
                "temperature",
                0
            )
        )

        predicted_hardware_temp = float(
            server.get(
                "predicted_temperature",
                temperature
            )
        )

        cooling = bool(
            server.get(
                "cooling_active",
                False
            )
        )

        stress = float(
            server.get(
                "thermal_stress",
                0
            )
        )

        features, predictions = predict_server(
            server
        )

        future_temp = predictions.get(
            "future_temperature",
            predicted_hardware_temp
        )

        risk = predictions.get(
            "hotspot_risk",
            0
        )

        required_fan = predictions.get(
            "required_fan",
            0
        )

        cooling_effectiveness = predictions.get(
            "cooling_effectiveness",
            0
        )

        overheating_warning = predictions.get(
            "overheating_warning",
            0
        )

        status = get_status(
            temperature
        )

        hotspot = get_hotspot_status(
            temperature,
            future_temp,
            cooling
        )

        protection = get_protection_status(
            temperature
        )

        # Make high temperature visible even if the
        # ML risk percentage happens to be low.
        display_risk = max(
            float(risk),
            100 if temperature >= 50 else
            80 if temperature >= 40 else
            35 if temperature >= 35 else
            0
        )

        rows.append({
            "Server": server_id,
            "Temperature": temperature,
            "Predicted Temperature": predicted_hardware_temp,
            "AI Future Temperature": future_temp,
            "Status": status,
            "Cooling": cooling,
            "Hotspot": hotspot,
            "Protection": protection,
            "Stress": stress,
            "Risk": display_risk,
            "Required Fan": required_fan,
            "Cooling Effectiveness": cooling_effectiveness,
            "Overheating Warning": overheating_warning,
            "Features": features,
            "Predictions": predictions
        })


    df = pd.DataFrame(
        rows
    )


    # =====================================================
    # TOP LIVE STATUS
    # =====================================================

    st.markdown(
        '<div class="section-title">📡 Live System Status</div>',
        unsafe_allow_html=True
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    k1.metric(
        "Servers Online",
        len(df)
    )

    k2.metric(
        "Highest Temperature",
        f"{df['Temperature'].max():.1f} °C"
    )

    k3.metric(
        "Average Temperature",
        f"{df['Temperature'].mean():.1f} °C"
    )

    k4.metric(
        "Critical Servers",
        int(
            (df["Temperature"] >= 50).sum()
        )
    )

    k5.metric(
        "Cooling Active",
        int(
            df["Cooling"].sum()
        )
    )

    k6.metric(
        "MQTT",
        "LIVE"
        if mqtt_system.connected
        else "OFFLINE"
    )


    # =====================================================
    # SERVER READING CARDS
    # =====================================================

    st.markdown(
        '<div class="section-title">🖥️ Real-Time Server Readings</div>',
        unsafe_allow_html=True
    )

    card_columns = st.columns(
        min(4, len(rows))
    )

    for index, row in enumerate(rows):

        with card_columns[
            index % len(card_columns)
        ]:

            temp = row["Temperature"]

            if temp >= 50:

                st.markdown(
                    f"""
                    <div class="critical-box">
                    <h3>🔴 Server {row['Server']}</h3>
                    <b>Temperature:</b> {temp:.2f} °C<br>
                    <b>Status:</b> CRITICAL<br>
                    <b>Hotspot:</b> {row['Hotspot']}<br>
                    <b>Cooling:</b> {'ON' if row['Cooling'] else 'OFF'}<br>
                    <b>Protection:</b> {row['Protection']}<br>
                    <b>Thermal Stress:</b> {row['Stress']:.1f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif temp >= 40:

                st.markdown(
                    f"""
                    <div class="warning-box">
                    <h3>🟠 Server {row['Server']}</h3>
                    <b>Temperature:</b> {temp:.2f} °C<br>
                    <b>Status:</b> HIGH TEMPERATURE<br>
                    <b>Hotspot:</b> {row['Hotspot']}<br>
                    <b>Cooling:</b> {'ON' if row['Cooling'] else 'OFF'}<br>
                    <b>Protection:</b> {row['Protection']}<br>
                    <b>Thermal Stress:</b> {row['Stress']:.1f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="safe-box">
                    <h3>🟢 Server {row['Server']}</h3>
                    <b>Temperature:</b> {temp:.2f} °C<br>
                    <b>Status:</b> {row['Status']}<br>
                    <b>Hotspot:</b> {row['Hotspot']}<br>
                    <b>Cooling:</b> {'ON' if row['Cooling'] else 'OFF'}<br>
                    <b>Protection:</b> {row['Protection']}<br>
                    <b>Thermal Stress:</b> {row['Stress']:.1f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


    # =====================================================
    # FULL READING TABLE
    # =====================================================

    st.markdown(
        '<div class="section-title">📋 Live Hardware Readings</div>',
        unsafe_allow_html=True
    )

    table = df[
        [
            "Server",
            "Temperature",
            "Predicted Temperature",
            "Status",
            "Cooling",
            "Hotspot",
            "Protection",
            "Stress"
        ]
    ].copy()

    table.columns = [
        "Server",
        "Current °C",
        "Wokwi Predicted °C",
        "Hardware Status",
        "Cooling",
        "Hotspot Status",
        "Protection",
        "Thermal Stress"
    ]

    st.dataframe(
        table.round(2),
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # SERVER DETAIL
    # =====================================================

    st.markdown(
        '<div class="section-title">🔍 Detailed Server Analysis</div>',
        unsafe_allow_html=True
    )

    selected_server = st.selectbox(
        "Select server",
        [
            int(x["Server"])
            for x in rows
        ]
    )

    selected = next(
        x for x in rows
        if int(x["Server"]) ==
        selected_server
    )

    features = selected["Features"]
    predictions = selected["Predictions"]


    a, b, c, d = st.columns(4)

    a.metric(
        "GPU Usage",
        f"{features['GPU_usage_percent']:.1f}%"
    )

    b.metric(
        "GPU Power",
        f"{features['GPU_power_W']:.1f} W"
    )

    c.metric(
        "CPU Usage",
        f"{features['CPU_usage_percent']:.1f}%"
    )

    d.metric(
        "CPU Temperature",
        f"{features['CPU_temperature_C']:.1f} °C"
    )


    a, b, c, d = st.columns(4)

    a.metric(
        "Ambient",
        f"{features['ambient_temperature_C']:.1f} °C"
    )

    b.metric(
        "Inlet",
        f"{features['inlet_temperature_C']:.1f} °C"
    )

    c.metric(
        "Outlet",
        f"{features['outlet_temperature_C']:.1f} °C"
    )

    d.metric(
        "Airflow",
        f"{features['airflow_m3_s']:.2f} m³/s"
    )


    a, b, c, d = st.columns(4)

    a.metric(
        "Fan Speed",
        f"{features['fan_speed_percent']:.1f}%"
    )

    b.metric(
        "Workload",
        f"{features['workload_percent']:.1f}%"
    )

    c.metric(
        "AI Future GPU Temp",
        f"{predictions.get('future_temperature', 0):.2f} °C"
    )

    d.metric(
        "Required Fan",
        f"{predictions.get('required_fan', 0):.1f}%"
    )


    # =====================================================
    # AI RESULTS
    # =====================================================

    st.markdown(
        '<div class="section-title">🤖 AI Model Results</div>',
        unsafe_allow_html=True
    )

    ai1, ai2, ai3, ai4 = st.columns(4)

    ai1.metric(
        "Hotspot Risk",
        f"{max(selected['Risk'], 0):.1f}%"
    )

    ai2.metric(
        "Cooling Effectiveness",
        f"{predictions.get('cooling_effectiveness', 0):.1f}%"
    )

    ai3.metric(
        "Future Temperature",
        f"{predictions.get('future_temperature', 0):.2f} °C"
    )

    ai4.metric(
        "Overheating Warning",
        "YES"
        if predictions.get(
            "overheating_warning",
            0
        ) == 1
        else "NO"
    )


    # =====================================================
    # HIGHEST RISK SERVER
    # =====================================================

    highest = df.loc[
        df["Risk"].idxmax()
    ]

    st.markdown(
        '<div class="section-title">⚠️ Hardware Risk Analysis</div>',
        unsafe_allow_html=True
    )

    if highest["Temperature"] >= 50:

        st.error(
            f"Server {int(highest['Server'])} is currently at "
            f"{highest['Temperature']:.2f}°C and requires critical thermal protection."
        )

    elif highest["Temperature"] >= 40:

        st.warning(
            f"Server {int(highest['Server'])} is currently at "
            f"{highest['Temperature']:.2f}°C and is in the high-temperature range."
        )

    elif highest["Temperature"] >= 35:

        st.warning(
            f"Server {int(highest['Server'])} is currently at "
            f"{highest['Temperature']:.2f}°C and should be monitored."
        )

    else:

        st.success(
            f"All servers are currently below the high-temperature range. "
            f"Server {int(highest['Server'])} has the highest modeled risk."
        )


    # =====================================================
    # GRAPHS AT THE END
    # =====================================================

    st.markdown(
        '<div class="section-title">📊 Thermal Analytics</div>',
        unsafe_allow_html=True
    )


    # Temperature graph

    fig_temp = px.bar(
        df,
        x="Server",
        y="Temperature",
        text="Temperature",
        title="Current Server Temperatures"
    )

    fig_temp.add_hline(
        y=35,
        line_dash="dash",
        annotation_text="SAFE LIMIT"
    )

    fig_temp.add_hline(
        y=40,
        line_dash="dash",
        annotation_text="HIGH TEMPERATURE"
    )

    fig_temp.add_hline(
        y=50,
        line_dash="dash",
        annotation_text="CRITICAL"
    )

    fig_temp.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        height=430
    )

    st.plotly_chart(
        fig_temp,
        use_container_width=True
    )


    # Risk graph

    fig_risk = px.bar(
        df,
        x="Server",
        y="Risk",
        text="Risk",
        title="AI Hotspot Risk by Server"
    )

    fig_risk.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        height=430,
        yaxis_title="Risk %"
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True
    )


    # Cooling graph

    cooling_df = df[
        [
            "Server",
            "Cooling Effectiveness"
        ]
    ]

    fig_cooling = px.bar(
        cooling_df,
        x="Server",
        y="Cooling Effectiveness",
        text="Cooling Effectiveness",
        title="Cooling Effectiveness"
    )

    fig_cooling.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        height=430,
        yaxis_title="Effectiveness %"
    )

    st.plotly_chart(
        fig_cooling,
        use_container_width=True
    )


    # =====================================================
    # MODEL PERFORMANCE
    # =====================================================

    st.markdown(
        '<div class="section-title">🧠 Model Performance</div>',
        unsafe_allow_html=True
    )

    performance = []

    for model_name, values in model_metrics.items():

        if "accuracy" in values:

            performance.append({
                "Model": model_name,
                "Metric": "Accuracy",
                "Value": f"{values['accuracy'] * 100:.2f}%"
            })

        else:

            performance.append({
                "Model": model_name,
                "Metric": "R²",
                "Value": f"{values['r2'] * 100:.2f}%"
            })

            performance.append({
                "Model": model_name,
                "Metric": "MAE",
                "Value": f"{values['mae']:.2f}"
            })

    st.dataframe(
        pd.DataFrame(performance),
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # DATA NOTE
    # =====================================================

    st.info(
        "Prototype note: Wokwi currently provides the live temperature, "
        "cooling, prediction and thermal-stress readings. GPU, CPU, power, "
        "airflow and workload values are estimated because those physical "
        "sensors are not currently present in the four-server Wokwi circuit. "
        "The five ML training datasets are synthetic prototype datasets."
    )


# =========================================================
# RUN
# =========================================================

realtime_dashboard()
