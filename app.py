import json
import threading
import time
import os

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import paho.mqtt.client as mqtt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Server Hotspot Cooling",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# MQTT CONFIG
# =========================================================

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "ai-cooling/WOKWI-ESP32-001"


# =========================================================
# ML FEATURES
# =========================================================

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
    "Model 1 - Future GPU Temperature":
        ("server_cooling_model1_1000_rows.csv",
         "future_GPU_temperature_C"),

    "Model 2 - Required Fan Speed":
        ("server_cooling_model2_1000_rows.csv",
         "required_fan_speed_percent"),

    "Model 3 - Hotspot Risk":
        ("server_cooling_model3_1000_rows.csv",
         "hotspot_risk_percent"),

    "Model 4 - Cooling Effectiveness":
        ("server_cooling_model4_1000_rows.csv",
         "cooling_effectiveness_percent"),

    "Model 5 - Overheating Warning":
        ("server_cooling_model5_1000_rows.csv",
         "overheating_warning")
}


# =========================================================
# LIGHT UI
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

.main-title {
    font-size: 34px;
    font-weight: 750;
    color: #172033;
    margin-bottom: 3px;
}

.subtitle {
    color: #667085;
    font-size: 15px;
    margin-bottom: 20px;
}

.section-title {
    font-size: 23px;
    font-weight: 700;
    color: #172033;
    margin-top: 18px;
    margin-bottom: 12px;
}

.kpi-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
}

.live-box {
    padding: 12px 16px;
    border-radius: 10px;
    border: 1px solid #d9d9d9;
    background: white;
}

.server-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
}

.small-text {
    color: #667085;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MQTT LIVE DATA
# =========================================================

@st.cache_resource
def get_mqtt_system():

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

            try:
                self.client.connect(
                    MQTT_BROKER,
                    MQTT_PORT,
                    60
                )

                self.client.loop_start()

            except Exception:
                self.connected = False

        def on_connect(self, client, userdata, flags, reason_code, properties):
            if reason_code == 0:
                self.connected = True
                client.subscribe(MQTT_TOPIC)

        def on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
            self.connected = False

        def on_message(self, client, userdata, msg):

            try:
                payload = json.loads(
                    msg.payload.decode("utf-8")
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


mqtt_system = get_mqtt_system()


# =========================================================
# LOAD AND TRAIN MODELS
# =========================================================

@st.cache_resource
def load_models():

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
            test_size=0.2,
            random_state=42
        )

        if target == "overheating_warning":

            model = RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                max_depth=12
            )

            model.fit(X_train, y_train)

            prediction = model.predict(X_test)

            metrics[model_name] = {
                "accuracy": accuracy_score(
                    y_test,
                    prediction
                )
            }

        else:

            model = RandomForestRegressor(
                n_estimators=150,
                random_state=42,
                max_depth=12
            )

            model.fit(X_train, y_train)

            prediction = model.predict(X_test)

            metrics[model_name] = {
                "r2": r2_score(
                    y_test,
                    prediction
                ),
                "mae": mean_absolute_error(
                    y_test,
                    prediction
                )
            }

        models[model_name] = model

    return models, metrics


models, model_metrics = load_models()


# =========================================================
# ESTIMATE EXTRA SERVER PARAMETERS
# =========================================================

def create_server_features(server):

    temp = float(
        server.get("temperature", 25)
    )

    cooling = bool(
        server.get("cooling_active", False)
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

    inlet = ambient + max(
        0,
        temp - 35
    ) * 0.08

    outlet = inlet + max(
        0,
        temp - 30
    ) * 0.18

    fan_speed = 75 if cooling else 35

    airflow = 1.0 + fan_speed / 100

    workload = (
        gpu_usage + cpu_usage
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
# RUN ALL FIVE MODELS
# =========================================================

def run_predictions(server):

    features = create_server_features(server)

    X = pd.DataFrame(
        [features],
        columns=FEATURES
    )

    results = {}

    for model_name, model in models.items():

        try:

            prediction = model.predict(X)[0]

            if "Overheating Warning" in model_name:

                results["overheating_warning"] = int(
                    prediction
                )

            elif "Future GPU" in model_name:

                results["future_temperature"] = float(
                    prediction
                )

            elif "Required Fan" in model_name:

                results["required_fan"] = float(
                    prediction
                )

            elif "Hotspot Risk" in model_name:

                results["hotspot_risk"] = float(
                    prediction
                )

            elif "Cooling Effectiveness" in model_name:

                results["cooling_effectiveness"] = float(
                    prediction
                )

        except Exception:

            pass

    return features, results


# =========================================================
# TEMPERATURE STATUS
# =========================================================

def temperature_status(temp):

    if temp >= 50:
        return "CRITICAL"

    if temp >= 40:
        return "HIGH TEMPERATURE"

    if temp >= 35:
        return "WARM - MONITORING"

    return "SAFE"


def status_level(temp):

    if temp >= 50:
        return 3

    if temp >= 40:
        return 2

    if temp >= 35:
        return 1

    return 0


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">❄️ AI Server Hotspot Cooling System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Real-time Wokwi thermal monitoring with five machine-learning models</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("System Control")

    device_id = st.text_input(
        "Device ID",
        "WOKWI-ESP32-001"
    )

    server_count = st.number_input(
        "Number of Servers",
        min_value=1,
        max_value=20,
        value=4
    )

    st.divider()

    st.subheader("Connection")

    if mqtt_system.connected:

        st.success("MQTT Connected")

    else:

        st.error("MQTT Disconnected")

    st.caption(
        f"Broker: {MQTT_BROKER}"
    )

    st.caption(
        f"Topic: {MQTT_TOPIC}"
    )

    st.divider()

    st.subheader("Temperature Limits")

    st.write("🟢 Safe: < 35°C")
    st.write("🟡 Warm: 35–39.9°C")
    st.write("🟠 High: 40–49.9°C")
    st.write("🔴 Critical: ≥ 50°C")


# =========================================================
# LIVE DASHBOARD
# =========================================================

@st.fragment(run_every=3)
def live_dashboard():

    live_data = mqtt_system.get_data()

    if live_data is None:

        st.warning(
            "Waiting for live Wokwi MQTT data..."
        )

        st.info(
            "Start your Wokwi simulation and make sure MQTT publishing is active."
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
            "No server data received."
        )

        return


    # =====================================================
    # PROCESS SERVERS
    # =====================================================

    processed = []

    for server in servers:

        features, predictions = run_predictions(
            server
        )

        temp = float(
            server.get(
                "temperature",
                0
            )
        )

        status = temperature_status(
            temp
        )

        risk = predictions.get(
            "hotspot_risk",
            0
        )

        future_temp = predictions.get(
            "future_temperature",
            temp
        )

        warning = predictions.get(
            "overheating_warning",
            0
        )

        processed.append({
            "server_id": server.get(
                "server_id",
                len(processed) + 1
            ),
            "temperature": temp,
            "status": status,
            "cooling": server.get(
                "cooling_active",
                False
            ),
            "hotspot": server.get(
                "hotspot_status",
                "NO HOTSPOT"
            ),
            "stress": float(
                server.get(
                    "thermal_stress",
                    0
                )
            ),
            "future_temperature": future_temp,
            "risk": risk,
            "warning": warning,
            "features": features,
            "predictions": predictions
        })


    df = pd.DataFrame(
        processed
    )


    # =====================================================
    # KPI SECTION
    # =====================================================

    total_servers = len(df)

    highest_temp = df["temperature"].max()

    average_temp = df["temperature"].mean()

    critical_count = (
        df["temperature"] >= 50
    ).sum()

    cooling_count = (
        df["cooling"] == True
    ).sum()

    safe_count = (
        df["temperature"] < 35
    ).sum()


    st.markdown(
        '<div class="section-title">Live System Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric(
        "Servers",
        total_servers
    )

    c2.metric(
        "Highest Temp",
        f"{highest_temp:.1f} °C"
    )

    c3.metric(
        "Average Temp",
        f"{average_temp:.1f} °C"
    )

    c4.metric(
        "Critical",
        critical_count
    )

    c5.metric(
        "Cooling Active",
        cooling_count
    )

    c6.metric(
        "Safe",
        safe_count
    )


    # =====================================================
    # HARDWARE STATUS TABLE
    # =====================================================

    st.markdown(
        '<div class="section-title">Real-Time Hardware Status</div>',
        unsafe_allow_html=True
    )

    status_table = df[
        [
            "server_id",
            "temperature",
            "status",
            "cooling",
            "stress",
            "risk"
        ]
    ].copy()

    status_table.columns = [
        "Server",
        "Temperature °C",
        "Hardware Status",
        "Cooling",
        "Thermal Stress",
        "Hotspot Risk %"
    ]

    status_table[
        "Temperature °C"
    ] = status_table[
        "Temperature °C"
    ].round(2)

    status_table[
        "Hotspot Risk %"
    ] = status_table[
        "Hotspot Risk %"
    ].round(2)

    st.dataframe(
        status_table,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # CHARTS
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Server Temperature"
        )

        fig_temp = px.bar(
            df,
            x="server_id",
            y="temperature",
            text="temperature",
            labels={
                "server_id": "Server",
                "temperature": "Temperature °C"
            }
        )

        fig_temp.add_hline(
            y=35,
            line_dash="dash",
            annotation_text="Safe limit"
        )

        fig_temp.add_hline(
            y=40,
            line_dash="dash",
            annotation_text="Warning"
        )

        fig_temp.add_hline(
            y=50,
            line_dash="dash",
            annotation_text="Critical"
        )

        fig_temp.update_layout(
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig_temp,
            use_container_width=True
        )


    with col2:

        st.subheader(
            "Thermal Health Distribution"
        )

        safe = int(
            (df["temperature"] < 35).sum()
        )

        warm = int(
            ((df["temperature"] >= 35) &
             (df["temperature"] < 40)).sum()
        )

        high = int(
            ((df["temperature"] >= 40) &
             (df["temperature"] < 50)).sum()
        )

        critical = int(
            (df["temperature"] >= 50).sum()
        )

        health_df = pd.DataFrame({
            "Status": [
                "Safe",
                "Warm",
                "High",
                "Critical"
            ],
            "Servers": [
                safe,
                warm,
                high,
                critical
            ]
        })

        fig_health = px.pie(
            health_df,
            names="Status",
            values="Servers",
            hole=0.55
        )

        fig_health.update_layout(
            height=400,
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig_health,
            use_container_width=True
        )


    # =====================================================
    # ML PREDICTIONS
    # =====================================================

    st.markdown(
        '<div class="section-title">AI / ML Predictions</div>',
        unsafe_allow_html=True
    )

    ml_table = []

    for row in processed:

        p = row["predictions"]

        ml_table.append({
            "Server":
                f"Server {row['server_id']}",

            "Current °C":
                round(row["temperature"], 2),

            "Future GPU °C":
                round(
                    p.get(
                        "future_temperature",
                        0
                    ),
                    2
                ),

            "Required Fan %":
                round(
                    p.get(
                        "required_fan",
                        0
                    ),
                    2
                ),

            "Hotspot Risk %":
                round(
                    p.get(
                        "hotspot_risk",
                        0
                    ),
                    2
                ),

            "Cooling Effectiveness %":
                round(
                    p.get(
                        "cooling_effectiveness",
                        0
                    ),
                    2
                ),

            "Overheating Warning":
                "⚠️ YES"
                if p.get(
                    "overheating_warning",
                    0
                ) == 1
                else "✅ NO"
        })

    ml_df = pd.DataFrame(
        ml_table
    )

    st.dataframe(
        ml_df,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # SERVER AT RISK
    # =====================================================

    st.markdown(
        '<div class="section-title">Hardware Damage Risk Monitoring</div>',
        unsafe_allow_html=True
    )

    risk_df = df[
        [
            "server_id",
            "temperature",
            "future_temperature",
            "risk",
            "stress"
        ]
    ].copy()

    risk_df = risk_df.sort_values(
        "risk",
        ascending=False
    )

    risk_df.columns = [
        "Server",
        "Current °C",
        "Predicted Future °C",
        "Hotspot Risk %",
        "Thermal Stress"
    ]

    st.dataframe(
        risk_df.round(2),
        use_container_width=True,
        hide_index=True
    )

    highest_risk = risk_df.iloc[0]

    if highest_risk["Hotspot Risk %"] >= 70:

        st.error(
            f"⚠️ Server {int(highest_risk['Server'])} currently has the highest modeled hotspot risk."
        )

    elif highest_risk["Hotspot Risk %"] >= 40:

        st.warning(
            f"⚠️ Server {int(highest_risk['Server'])} requires thermal monitoring."
        )

    else:

        st.success(
            "All monitored servers currently have relatively low modeled hotspot risk."
        )


    # =====================================================
    # SERVER DETAIL
    # =====================================================

    st.markdown(
        '<div class="section-title">Server Detailed Analysis</div>',
        unsafe_allow_html=True
    )

    selected_server = st.selectbox(
        "Select Server",
        [
            int(x["server_id"])
            for x in processed
        ]
    )

    selected = next(
        x for x in processed
        if int(x["server_id"]) ==
        selected_server
    )

    features = selected["features"]
    predictions = selected["predictions"]

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
        "Inlet Temperature",
        f"{features['inlet_temperature_C']:.1f} °C"
    )

    b.metric(
        "Outlet Temperature",
        f"{features['outlet_temperature_C']:.1f} °C"
    )

    c.metric(
        "Fan Speed",
        f"{features['fan_speed_percent']:.1f}%"
    )

    d.metric(
        "Airflow",
        f"{features['airflow_m3_s']:.2f} m³/s"
    )


    # =====================================================
    # MODEL 5 WARNING
    # =====================================================

    if predictions.get(
        "overheating_warning",
        0
    ) == 1:

        st.error(
            "🚨 AI overheating warning detected for this server."
        )

    else:

        st.success(
            "✅ AI overheating model currently reports no overheating warning."
        )


    # =====================================================
    # DATA SOURCE NOTE
    # =====================================================

    st.info(
        "The Wokwi ESP32 currently provides live temperature, cooling and thermal-stress data. "
        "The additional GPU/CPU/power/airflow parameters are estimated from the simulated "
        "temperature because those physical sensors are not currently present in the Wokwi circuit. "
        "The five training datasets are synthetic prototype data."
    )


# =========================================================
# RUN DASHBOARD
# =========================================================

live_dashboard()
