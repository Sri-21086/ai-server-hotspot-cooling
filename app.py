import streamlit as st
import paho.mqtt.client as mqtt
import json
import threading
import time
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


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
# LIGHT THEME / STYLE
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
}

.status-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
}

.small-muted {
    color: #667085;
    font-size: 13px;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e5e7eb;
    padding: 12px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MQTT SETTINGS
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


# =========================================================
# PERSISTENT MQTT SYSTEM
# =========================================================

@st.cache_resource
def get_mqtt_system():

    system = {
        "client": None,
        "data": {},
        "history": [],
        "last_update": None,
        "connected": False,
        "lock": threading.Lock()
    }

    def on_connect(client, userdata, flags, reason_code, properties=None):

        if reason_code == 0:

            system["connected"] = True

            client.subscribe(MQTT_TOPIC)

            print("MQTT subscribed:", MQTT_TOPIC)

        else:

            system["connected"] = False

            print("MQTT connection failed:", reason_code)

    def on_disconnect(
        client,
        userdata,
        disconnect_flags,
        reason_code,
        properties=None
    ):

        system["connected"] = False

        print("MQTT disconnected:", reason_code)

    def on_message(client, userdata, message):

        try:

            payload = json.loads(
                message.payload.decode("utf-8")
            )

            servers = payload.get(
                "servers",
                []
            )

            timestamp = time.time()

            with system["lock"]:

                system["data"] = payload
                system["last_update"] = timestamp

                for server in servers:

                    server_copy = dict(server)

                    server_copy["timestamp"] = timestamp

                    system["history"].append(
                        server_copy
                    )

                # Keep last 300 readings
                if len(system["history"]) > 300:

                    system["history"] = (
                        system["history"][-300:]
                    )

        except Exception as e:

            print("MQTT message error:", e)

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="ai-cooling-dashboard-001"
    )

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    system["client"] = client

    return system


mqtt_system = get_mqtt_system()


# =========================================================
# CONNECT / DISCONNECT
# =========================================================

def connect_mqtt():

    try:

        if mqtt_system["client"].is_connected():

            mqtt_system["connected"] = True

            return True

        mqtt_system["client"].connect(
            MQTT_BROKER,
            MQTT_PORT,
            60
        )

        mqtt_system["client"].loop_start()

        return True

    except Exception as e:

        st.error(
            f"MQTT connection failed: {e}"
        )

        return False


def disconnect_mqtt():

    try:

        mqtt_system["client"].loop_stop()

        mqtt_system["client"].disconnect()

        mqtt_system["connected"] = False

    except Exception:

        pass


# =========================================================
# TRAIN ML MODELS
# =========================================================

@st.cache_resource
def train_models():

    models = {}

    datasets = {
        "model1": (
            "server_cooling_model1_1000_rows.csv",
            "future_GPU_temperature_C",
            "regression"
        ),
        "model2": (
            "server_cooling_model2_1000_rows.csv",
            "required_fan_speed_percent",
            "regression"
        ),
        "model3": (
            "server_cooling_model3_1000_rows.csv",
            "hotspot_risk_percent",
            "regression"
        ),
        "model4": (
            "server_cooling_model4_1000_rows.csv",
            "cooling_effectiveness_percent",
            "regression"
        ),
        "model5": (
            "server_cooling_model5_1000_rows.csv",
            "overheating_warning",
            "classification"
        )
    }

    for name, (
        filename,
        target,
        model_type
    ) in datasets.items():

        if not os.path.exists(filename):

            continue

        try:

            df = pd.read_csv(filename)

            available_features = [
                feature
                for feature in FEATURES
                if feature in df.columns
            ]

            if not available_features:
                continue

            if target not in df.columns:
                continue

            X = df[available_features]
            y = df[target]

            if model_type == "classification":

                model = RandomForestClassifier(
                    n_estimators=120,
                    random_state=42,
                    class_weight="balanced"
                )

            else:

                model = RandomForestRegressor(
                    n_estimators=120,
                    random_state=42
                )

            model.fit(X, y)

            models[name] = (
                model,
                available_features
            )

        except Exception as e:

            print(
                f"Model {name} error:",
                e
            )

    return models


models = train_models()


# =========================================================
# CREATE ML INPUT FROM WOKWI DATA
# =========================================================

def create_ml_input(server):

    temperature = float(
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

    predicted = float(
        server.get(
            "predicted_temperature",
            temperature
        )
    )

    # These are estimated simulation inputs because
    # the current Wokwi hardware provides temperature
    # rather than real GPU/CPU telemetry.

    gpu_usage = min(
        max((temperature - 20) * 2, 0),
        100
    )

    cpu_usage = min(
        max((temperature - 20) * 1.8, 0),
        100
    )

    fan_speed = (
        90 if cooling else 30
    )

    data = {

        "GPU_usage_percent":
            gpu_usage,

        "GPU_power_W":
            min(
                max((temperature - 20) * 8, 0),
                500
            ),

        "CPU_usage_percent":
            cpu_usage,

        "CPU_temperature_C":
            max(
                temperature - 2,
                20
            ),

        "GPU_temperature_C":
            temperature,

        "ambient_temperature_C":
            25,

        "inlet_temperature_C":
            max(
                temperature - 5,
                15
            ),

        "outlet_temperature_C":
            temperature + 2,

        "fan_speed_percent":
            fan_speed,

        "airflow_m3_s":
            1.0 + fan_speed / 100,

        "workload_percent":
            gpu_usage
    }

    return pd.DataFrame([data])


# =========================================================
# RUN ML
# =========================================================

def run_ml(server):

    input_df = create_ml_input(server)

    temperature = float(
        server.get(
            "temperature",
            25
        )
    )

    results = {}

    # -----------------------------
    # MODEL 1
    # -----------------------------

    if "model1" in models:

        model, features = models["model1"]

        results["future_temperature"] = float(
            model.predict(
                input_df[features]
            )[0]
        )

    else:

        results["future_temperature"] = float(
            server.get(
                "predicted_temperature",
                temperature
            )
        )

    # -----------------------------
    # MODEL 2
    # -----------------------------

    if "model2" in models:

        model, features = models["model2"]

        results["fan_speed"] = float(
            model.predict(
                input_df[features]
            )[0]
        )

    else:

        results["fan_speed"] = (
            100 if temperature >= 50
            else 75 if temperature >= 40
            else 30
        )

    # -----------------------------
    # MODEL 3
    # -----------------------------

    if "model3" in models:

        model, features = models["model3"]

        results["hotspot_risk"] = float(
            model.predict(
                input_df[features]
            )[0]
        )

    else:

        results["hotspot_risk"] = min(
            max(
                (temperature - 30) * 5,
                0
            ),
            100
        )

    # -----------------------------
    # MODEL 4
    # -----------------------------

    if "model4" in models:

        model, features = models["model4"]

        results["cooling_effectiveness"] = float(
            model.predict(
                input_df[features]
            )[0]
        )

    else:

        results["cooling_effectiveness"] = (
            85 if server.get(
                "cooling_active",
                False
            )
            else 30
        )

    # -----------------------------
    # MODEL 5
    # -----------------------------

    # Model 5 was not successfully
    # trained as a classification
    # model in AutoTrain.
    #
    # Therefore use the validated
    # temperature rule here.

    results["overheating_warning"] = int(
        temperature >= 50
    )

    return results


# =========================================================
# RISK CALCULATION
# =========================================================

def calculate_risk(server, ml):

    temperature = float(
        server.get(
            "temperature",
            0
        )
    )

    stress = float(
        server.get(
            "thermal_stress",
            0
        )
    )

    hotspot = float(
        ml.get(
            "hotspot_risk",
            0
        )
    )

    score = (
        min(temperature / 50 * 50, 50)
        +
        min(stress / 10 * 20, 20)
        +
        min(hotspot / 100 * 30, 30)
    )

    return min(
        max(score, 0),
        100
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '❄️ AI Server Hotspot Cooling System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Real-time thermal monitoring • AI prediction • hotspot risk • cooling protection'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🔌 Device Connection")

    device_id = st.text_input(
        "Device ID",
        value="WOKWI-ESP32-001"
    )

    server_count = st.number_input(
        "Dashboard Server Count",
        min_value=1,
        max_value=20,
        value=4,
        step=1
    )

    st.caption(
        "Your current Wokwi simulation contains 4 servers."
    )

    if "user_connected" not in st.session_state:

        st.session_state.user_connected = False

    button_text = (
        "🔴 Disconnect"
        if st.session_state.user_connected
        else "🟢 Connect"
    )

    if st.button(
        button_text,
        use_container_width=True
    ):

        if not st.session_state.user_connected:

            if connect_mqtt():

                st.session_state.user_connected = True

        else:

            disconnect_mqtt()

            st.session_state.user_connected = False

    st.divider()

    if (
        st.session_state.user_connected
        and mqtt_system["connected"]
    ):

        st.success(
            "🟢 MQTT connection active"
        )

    else:

        st.error(
            "🔴 MQTT disconnected"
        )

    st.caption(
        "Live dashboard refresh: every 3 seconds"
    )


# =========================================================
# DASHBOARD FUNCTION
# =========================================================

@st.fragment(run_every=3)
def live_dashboard():

    # -----------------------------------------------------
    # READ LATEST DATA
    # -----------------------------------------------------

    with mqtt_system["lock"]:

        live_data = dict(
            mqtt_system["data"]
        )

        history = list(
            mqtt_system["history"]
        )

        last_update = (
            mqtt_system["last_update"]
        )

        mqtt_connected = (
            mqtt_system["connected"]
        )

    # -----------------------------------------------------
    # CONNECTION
    # -----------------------------------------------------

    if not mqtt_connected:

        st.warning(
            "⚠️ MQTT is currently disconnected."
        )

        return

    # -----------------------------------------------------
    # WAIT FOR DATA
    # -----------------------------------------------------

    if not live_data:

        st.info(
            "⏳ Connected successfully. Waiting for Wokwi data..."
        )

        return

    servers = live_data.get(
        "servers",
        []
    )

    received_device = live_data.get(
        "device_id",
        device_id
    )

    # -----------------------------------------------------
    # LAST UPDATE
    # -----------------------------------------------------

    age = 0

    if last_update:

        age = time.time() - last_update

    if age <= 6:

        st.success(
            f"🟢 LIVE • Last Wokwi update {age:.1f}s ago"
        )

    elif age <= 12:

        st.warning(
            f"🟡 DELAYED • Last update {age:.1f}s ago"
        )

    else:

        st.error(
            f"🔴 STALE DATA • Last update {age:.1f}s ago"
        )

    # =====================================================
    # TOP KPI ROW
    # =====================================================

    total_servers = len(servers)

    critical_servers = 0
    warning_servers = 0
    safe_servers = 0
    cooling_servers = 0

    temperatures = []

    for server in servers:

        temp = float(
            server.get(
                "temperature",
                0
            )
        )

        temperatures.append(temp)

        if temp >= 50:

            critical_servers += 1

        elif temp >= 35:

            warning_servers += 1

        else:

            safe_servers += 1

        if server.get(
            "cooling_active",
            False
        ):

            cooling_servers += 1

    highest_temp = (
        max(temperatures)
        if temperatures
        else 0
    )

    average_temp = (
        sum(temperatures) / len(temperatures)
        if temperatures
        else 0
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:

        st.metric(
            "🖥️ Servers",
            total_servers
        )

    with k2:

        st.metric(
            "🌡️ Highest Temp",
            f"{highest_temp:.1f} °C"
        )

    with k3:

        st.metric(
            "📊 Average Temp",
            f"{average_temp:.1f} °C"
        )

    with k4:

        st.metric(
            "🔴 Critical",
            critical_servers
        )

    with k5:

        st.metric(
            "❄️ Cooling Active",
            cooling_servers
        )

    with k6:

        st.metric(
            "🟢 Safe",
            safe_servers
        )

    st.divider()

    # =====================================================
    # SERVER STATUS TABLE
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '🖥️ Real-Time Hardware Status'
        '</div>',
        unsafe_allow_html=True
    )

    table_rows = []

    ml_cache = {}

    for index, server in enumerate(
        servers[:int(server_count)]
    ):

        server_id = server.get(
            "server_id",
            index + 1
        )

        ml = run_ml(server)

        ml_cache[str(server_id)] = ml

        temp = float(
            server.get(
                "temperature",
                0
            )
        )

        risk = calculate_risk(
            server,
            ml
        )

        if risk >= 70:

            health = "🔴 HIGH RISK"

        elif risk >= 40:

            health = "🟠 MONITOR"

        else:

            health = "🟢 NORMAL"

        table_rows.append({

            "Server":
                f"Server {server_id}",

            "Temperature":
                f"{temp:.2f} °C",

            "Future Temp":
                f"{ml['future_temperature']:.2f} °C",

            "Fan":
                f"{ml['fan_speed']:.0f}%",

            "Hotspot Risk":
                f"{ml['hotspot_risk']:.1f}%",

            "Cooling":
                "ON"
                if server.get(
                    "cooling_active",
                    False
                )
                else "OFF",

            "Protection":
                server.get(
                    "protection_status",
                    "NORMAL"
                ),

            "Thermal Health":
                health
        })

    if table_rows:

        st.dataframe(
            pd.DataFrame(table_rows),
            use_container_width=True,
            hide_index=True
        )

    # =====================================================
    # VISUAL ANALYTICS
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Thermal Analytics'
        '</div>',
        unsafe_allow_html=True
    )

    chart1, chart2 = st.columns(2)

    # -----------------------------------------------------
    # TEMPERATURE BAR CHART
    # -----------------------------------------------------

    with chart1:

        if servers:

            chart_df = pd.DataFrame({

                "Server": [
                    f"Server {s.get('server_id', i+1)}"
                    for i, s in enumerate(
                        servers[:int(server_count)]
                    )
                ],

                "Temperature": [
                    float(
                        s.get(
                            "temperature",
                            0
                        )
                    )
                    for s in servers[
                        :int(server_count)
                    ]
                ]
            })

            fig = px.bar(
                chart_df,
                x="Server",
                y="Temperature",
                title="Current Temperature by Server",
                text_auto=".1f"
            )

            fig.add_hline(
                y=35,
                line_dash="dash",
                annotation_text="Warm"
            )

            fig.add_hline(
                y=50,
                line_dash="dash",
                annotation_text="Critical"
            )

            fig.update_layout(
                height=360,
                plot_bgcolor="white",
                paper_bgcolor="white",
                yaxis_title="Temperature (°C)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # -----------------------------------------------------
    # STATUS DONUT
    # -----------------------------------------------------

    with chart2:

        status_df = pd.DataFrame({

            "Status": [
                "Safe",
                "Warning",
                "Critical"
            ],

            "Servers": [
                safe_servers,
                warning_servers,
                critical_servers
            ]
        })

        fig = px.pie(
            status_df,
            names="Status",
            values="Servers",
            hole=0.55,
            title="Current Server Health"
        )

        fig.update_layout(
            height=360,
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # LIVE TEMPERATURE HISTORY
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📈 Live Temperature Trend'
        '</div>',
        unsafe_allow_html=True
    )

    if history:

        history_df = pd.DataFrame(
            history
        )

        if "timestamp" in history_df.columns:

            history_df["Time"] = pd.to_datetime(
                history_df["timestamp"],
                unit="s"
            )

        if (
            "server_id" in history_df.columns
            and "temperature" in history_df.columns
        ):

            pivot = history_df.pivot_table(
                index="Time",
                columns="server_id",
                values="temperature",
                aggfunc="last"
            )

            pivot.columns = [
                f"Server {x}"
                for x in pivot.columns
            ]

            st.line_chart(
                pivot,
                height=380
            )

    # =====================================================
    # RISK / HARDWARE EXPOSURE
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '⚠️ Hardware Thermal Exposure'
        '</div>',
        unsafe_allow_html=True
    )

    risk_rows = []

    for index, server in enumerate(
        servers[:int(server_count)]
    ):

        server_id = server.get(
            "server_id",
            index + 1
        )

        ml = ml_cache[
            str(server_id)
        ]

        risk = calculate_risk(
            server,
            ml
        )

        if risk >= 70:

            level = "🔴 High"

        elif risk >= 40:

            level = "🟠 Moderate"

        else:

            level = "🟢 Low"

        risk_rows.append({

            "Server":
                f"Server {server_id}",

            "Thermal Risk":
                f"{risk:.1f}%",

            "Risk Level":
                level,

            "Temperature":
                f"{server.get('temperature', 0):.1f} °C",

            "Thermal Stress":
                f"{server.get('thermal_stress', 0):.1f}",

            "Hotspot Risk":
                f"{ml['hotspot_risk']:.1f}%",

            "Cooling":
                "Active"
                if server.get(
                    "cooling_active",
                    False
                )
                else "Inactive"
        })

    if risk_rows:

        risk_df = pd.DataFrame(
            risk_rows
        )

        st.dataframe(
            risk_df,
            use_container_width=True,
            hide_index=True
        )

    # =====================================================
    # SELECT SERVER
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '🔎 Detailed Server Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    options = [
        f"Server {s.get('server_id', i+1)}"
        for i, s in enumerate(
            servers[:int(server_count)]
        )
    ]

    if not options:

        return

    selected = st.selectbox(
        "Choose a server",
        options
    )

    selected_index = options.index(
        selected
    )

    selected_server = servers[
        selected_index
    ]

    selected_id = selected_server.get(
        "server_id",
        selected_index + 1
    )

    selected_ml = ml_cache[
        str(selected_id)
    ]

    # =====================================================
    # CURRENT SERVER
    # =====================================================

    st.markdown(
        f"### 🖥️ Server {selected_id}"
    )

    d1, d2, d3, d4 = st.columns(4)

    with d1:

        st.metric(
            "Current Temperature",
            f"{selected_server.get('temperature', 0):.2f} °C"
        )

    with d2:

        st.metric(
            "Future Temperature",
            f"{selected_ml['future_temperature']:.2f} °C"
        )

    with d3:

        st.metric(
            "Thermal Stress",
            f"{selected_server.get('thermal_stress', 0):.1f}"
        )

    with d4:

        st.metric(
            "Required Fan",
            f"{selected_ml['fan_speed']:.1f}%"
        )

    # =====================================================
    # EXTRA PARAMETERS
    # =====================================================

    st.markdown(
        "#### 🔧 Additional Thermal Parameters"
    )

    input_df = create_ml_input(
        selected_server
    )

    p1, p2, p3, p4, p5, p6 = st.columns(6)

    with p1:

        st.metric(
            "GPU Usage",
            f"{input_df.iloc[0]['GPU_usage_percent']:.1f}%"
        )

    with p2:

        st.metric(
            "CPU Usage",
            f"{input_df.iloc[0]['CPU_usage_percent']:.1f}%"
        )

    with p3:

        st.metric(
            "CPU Temp",
            f"{input_df.iloc[0]['CPU_temperature_C']:.1f} °C"
        )

    with p4:

        st.metric(
            "Inlet Temp",
            f"{input_df.iloc[0]['inlet_temperature_C']:.1f} °C"
        )

    with p5:

        st.metric(
            "Outlet Temp",
            f"{input_df.iloc[0]['outlet_temperature_C']:.1f} °C"
        )

    with p6:

        st.metric(
            "Airflow",
            f"{input_df.iloc[0]['airflow_m3_s']:.2f} m³/s"
        )

    # =====================================================
    # AI RESULTS
    # =====================================================

    st.markdown(
        "#### 🤖 AI Model Results"
    )

    a1, a2, a3, a4 = st.columns(4)

    with a1:

        st.metric(
            "Model 1",
            f"{selected_ml['future_temperature']:.2f} °C"
        )

        st.caption(
            "Future temperature"
        )

    with a2:

        st.metric(
            "Model 2",
            f"{selected_ml['fan_speed']:.1f}%"
        )

        st.caption(
            "Required fan speed"
        )

    with a3:

        st.metric(
            "Model 3",
            f"{selected_ml['hotspot_risk']:.1f}%"
        )

        st.caption(
            "Hotspot risk"
        )

    with a4:

        st.metric(
            "Model 4",
            f"{selected_ml['cooling_effectiveness']:.1f}%"
        )

        st.caption(
            "Cooling effectiveness"
        )

    # =====================================================
    # MODEL 5
    # =====================================================

    if selected_ml[
        "overheating_warning"
    ]:

        st.error(
            "🚨 OVERHEATING WARNING — "
            "temperature has reached the critical threshold."
        )

    else:

        st.success(
            "✅ No overheating warning"
        )

    # =====================================================
    # HARDWARE STATUS
    # =====================================================

    st.markdown(
        "#### 🛡️ Real-Time Protection Status"
    )

    current_temp = float(
        selected_server.get(
            "temperature",
            0
        )
    )

    if current_temp >= 50:

        st.error(
            "🔴 CRITICAL PROTECTION — "
            "Immediate thermal protection condition."
        )

    elif current_temp >= 40:

        st.warning(
            "🟠 HIGH TEMPERATURE — "
            "Protection response should remain active."
        )

    elif current_temp >= 35:

        st.warning(
            "🟡 WARM — "
            "Continuous monitoring recommended."
        )

    else:

        st.success(
            "🟢 NORMAL — "
            "Current temperature is within the safe range."
        )

    # =====================================================
    # WOKWI DATA
    # =====================================================

    with st.expander(
        "🔧 View Raw Wokwi / MQTT Data"
    ):

        st.json(
            selected_server
        )

    # =====================================================
    # FOOTER
    # =====================================================

    st.caption(
        "Live data source: Wokwi ESP32 → MQTT → Streamlit. "
        "Dashboard refreshes every 3 seconds."
    )


# =========================================================
# START LIVE DASHBOARD
# =========================================================

if st.session_state.get(
    "user_connected",
    False
):

    live_dashboard()

else:

    st.info(
        "🔌 Connect the Wokwi ESP32 to begin real-time monitoring."
    )

    st.markdown(
        """
        ### System Flow

        **Wokwi ESP32**
        → **MQTT**
        → **Live Streamlit Dashboard**
        → **AI/ML Predictions**
        → **Thermal Risk Analysis**
        → **Cooling Protection**
        """
    )
