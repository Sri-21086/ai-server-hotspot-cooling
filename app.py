import streamlit as st
import paho.mqtt.client as mqtt
import json
import threading
import time
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

st.set_page_config(
    page_title="AI Server Hotspot Cooling",
    page_icon="❄️",
    layout="wide"
)

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "ai-cooling/WOKWI-ESP32-001"

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

st.markdown("""
<style>
.main-title {
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 4px;
}

.subtitle {
    color: #777;
    font-size: 15px;
    margin-bottom: 20px;
}

.section-title {
    font-size: 22px;
    font-weight: 650;
    margin-top: 20px;
    margin-bottom: 12px;
}

.server-card {
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #dddddd;
    background: #fafafa;
    margin-bottom: 12px;
}

.live-box {
    padding: 12px 16px;
    border-radius: 10px;
    border: 1px solid #d9d9d9;
    background: #f7f7f7;
}

.small-text {
    color: #777;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# MQTT SYSTEM
# =========================================================

@st.cache_resource
def get_mqtt_system():

    system = {
        "client": None,
        "data": {},
        "last_update": None,
        "connected": False,
        "lock": threading.Lock()
    }

    def on_connect(client, userdata, flags, reason_code, properties=None):

        if reason_code == 0:

            system["connected"] = True

            client.subscribe(MQTT_TOPIC)

        else:

            system["connected"] = False

    def on_disconnect(
        client,
        userdata,
        disconnect_flags,
        reason_code,
        properties=None
    ):

        system["connected"] = False

    def on_message(client, userdata, message):

        try:

            payload = json.loads(
                message.payload.decode("utf-8")
            )

            with system["lock"]:

                system["data"] = payload
                system["last_update"] = time.time()

        except Exception as e:

            print("MQTT error:", e)

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="streamlit-dashboard-001"
    )

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    system["client"] = client

    return system


mqtt_system = get_mqtt_system()


def connect_mqtt():

    try:

        if not mqtt_system["connected"]:

            mqtt_system["client"].connect(
                MQTT_BROKER,
                MQTT_PORT,
                60
            )

            mqtt_system["client"].loop_start()

            time.sleep(1)

        return True

    except Exception as e:

        st.error(f"MQTT connection failed: {e}")

        return False


def disconnect_mqtt():

    try:

        if mqtt_system["connected"]:

            mqtt_system["client"].loop_stop()

            mqtt_system["client"].disconnect()

            mqtt_system["connected"] = False

    except Exception:

        pass


# =========================================================
# ML MODELS
# =========================================================

@st.cache_resource
def train_models():

    models = {}

    datasets = {
        "model1": (
            "server_cooling_model1_1000_rows.csv",
            "future_GPU_temperature_C"
        ),
        "model2": (
            "server_cooling_model2_1000_rows.csv",
            "required_fan_speed_percent"
        ),
        "model3": (
            "server_cooling_model3_1000_rows.csv",
            "hotspot_risk_percent"
        ),
        "model4": (
            "server_cooling_model4_1000_rows.csv",
            "cooling_effectiveness_percent"
        ),
        "model5": (
            "server_cooling_model5_1000_rows.csv",
            "overheating_warning"
        )
    }

    for name, (file, target) in datasets.items():

        if not os.path.exists(file):
            continue

        df = pd.read_csv(file)

        available_features = [
            x for x in FEATURES if x in df.columns
        ]

        if target not in df.columns:
            continue

        X = df[available_features]
        y = df[target]

        if name == "model5":

            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )

        else:

            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )

        model.fit(X, y)

        models[name] = (
            model,
            available_features
        )

    return models


models = train_models()


# =========================================================
# ML PREDICTION
# =========================================================

def create_input(server):

    temperature = float(
        server.get("temperature", 25)
    )

    predicted = float(
        server.get("predicted_temperature", temperature)
    )

    cooling = server.get(
        "cooling_active",
        False
    )

    fan_speed = 80 if cooling else 30

    input_data = {
        "GPU_usage_percent": min(
            max((temperature - 20) * 2, 0),
            100
        ),

        "GPU_power_W": min(
            max((temperature - 20) * 8, 0),
            500
        ),

        "CPU_usage_percent": min(
            max((temperature - 20) * 1.8, 0),
            100
        ),

        "CPU_temperature_C": temperature - 2,

        "GPU_temperature_C": temperature,

        "ambient_temperature_C": 25,

        "inlet_temperature_C": max(
            temperature - 5,
            15
        ),

        "outlet_temperature_C": temperature + 2,

        "fan_speed_percent": fan_speed,

        "airflow_m3_s": 1.0 + fan_speed / 100,

        "workload_percent": min(
            max((temperature - 20) * 2,
                0),
            100
        )
    }

    return pd.DataFrame([input_data])


def run_ml_predictions(server):

    input_df = create_input(server)

    results = {}

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
                server.get("temperature", 25)
            )
        )

    if "model2" in models:

        model, features = models["model2"]

        results["fan_speed"] = float(
            model.predict(
                input_df[features]
            )[0]
        )

    else:

        results["fan_speed"] = (
            100 if server.get("temperature", 0) >= 50
            else 70 if server.get("temperature", 0) >= 40
            else 30
        )

    if "model3" in models:

        model, features = models["model3"]

        results["hotspot_risk"] = float(
            model.predict(
                input_df[features]
            )[0]
        )

    else:

        temperature = server.get(
            "temperature",
            25
        )

        results["hotspot_risk"] = min(
            max((temperature - 30) * 5, 0),
            100
        )

    if "model4" in models:

        model, features = models["model4"]

        results["cooling_effectiveness"] = float(
            model.predict(
                input_df[features]
            )[0]
        )

    else:

        results["cooling_effectiveness"] = (
            85 if server.get("cooling_active")
            else 30
        )

    if "model5" in models:

        model, features = models["model5"]

        prediction = model.predict(
            input_df[features]
        )[0]

        results["overheating_warning"] = int(
            prediction
        )

    else:

        temperature = server.get(
            "temperature",
            25
        )

        results["overheating_warning"] = int(
            temperature >= 50
        )

    return results


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">❄️ AI Server Hotspot Cooling System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Real-time thermal monitoring • AI prediction • hotspot detection • cooling protection'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🔌 Device")

    device_id = st.text_input(
        "Device ID",
        "WOKWI-ESP32-001"
    )

    server_count = st.number_input(
        "Number of Servers",
        min_value=1,
        max_value=20,
        value=4,
        step=1
    )

    if "user_connected" not in st.session_state:

        st.session_state.user_connected = False

    if st.button(
        "Disconnect"
        if st.session_state.user_connected
        else "Connect",
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

        st.success("🟢 MQTT Connected")

    else:

        st.error("🔴 MQTT Disconnected")

    st.caption(
        "Data refresh interval: 3 seconds"
    )


# =========================================================
# MAIN DASHBOARD
# =========================================================

if not st.session_state.user_connected:

    st.info(
        "Connect the Wokwi device to start live monitoring."
    )

    st.stop()


# =========================================================
# LIVE DATA
# =========================================================

with mqtt_system["lock"]:

    live_data = dict(
        mqtt_system["data"]
    )

    last_update = mqtt_system["last_update"]


if not live_data:

    st.warning(
        "Waiting for MQTT data from Wokwi..."
    )

    st.caption(
        "Start the Wokwi simulation and wait for MQTT DATA SENT."
    )

    st.stop()


servers = live_data.get(
    "servers",
    []
)

received_device = live_data.get(
    "device_id",
    device_id
)


# =========================================================
# CONNECTION STATUS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Device",
        received_device
    )

with col2:

    st.metric(
        "Servers Received",
        len(servers)
    )

with col3:

    if last_update:

        age = time.time() - last_update

        st.metric(
            "Last Update",
            f"{age:.1f}s ago"
        )

with col4:

    st.metric(
        "Update Rate",
        "3 sec"
    )


st.divider()


# =========================================================
# SERVER OVERVIEW
# =========================================================

st.markdown(
    '<div class="section-title">🖥️ Server Overview</div>',
    unsafe_allow_html=True
)

actual_count = min(
    int(server_count),
    len(servers)
)

if actual_count == 0:

    st.warning(
        "No server data received."
    )

else:

    columns = st.columns(4)

    for i in range(actual_count):

        server = servers[i]

        server_id = server.get(
            "server_id",
            i + 1
        )

        temperature = float(
            server.get(
                "temperature",
                0
            )
        )

        status = server.get(
            "status",
            "UNKNOWN"
        )

        cooling = server.get(
            "cooling_active",
            False
        )

        with columns[i % 4]:

            st.markdown(
                f"### 🖥️ Server {server_id}"
            )

            st.metric(
                "Temperature",
                f"{temperature:.2f} °C"
            )

            if status == "SAFE":

                st.success("🟢 SAFE")

            elif status == "CRITICAL":

                st.error("🔴 CRITICAL")

            else:

                st.warning(
                    f"🟠 {status}"
                )

            if cooling:

                st.info("❄️ Cooling ON")

            else:

                st.write("Cooling OFF")


# =========================================================
# SERVER DETAILS
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">🔎 Detailed Server Analysis</div>',
    unsafe_allow_html=True
)

server_options = [
    f"Server {s.get('server_id', i + 1)}"
    for i, s in enumerate(
        servers[:actual_count]
    )
]

if not server_options:

    st.stop()

selected_server = st.selectbox(
    "Select Server",
    server_options
)

selected_index = server_options.index(
    selected_server
)

server = servers[selected_index]

server_id = server.get(
    "server_id",
    selected_index + 1
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

status = server.get(
    "status",
    "UNKNOWN"
)

cooling_active = server.get(
    "cooling_active",
    False
)

hotspot_status = server.get(
    "hotspot_status",
    "UNKNOWN"
)

thermal_stress = float(
    server.get(
        "thermal_stress",
        0
    )
)

protection_status = server.get(
    "protection_status",
    "UNKNOWN"
)


# =========================================================
# HARDWARE STATUS
# =========================================================

st.markdown(
    f"### 🖥️ Server {server_id}"
)

a, b, c = st.columns(3)

with a:

    st.metric(
        "🌡️ Current Temperature",
        f"{temperature:.2f} °C"
    )

with b:

    st.metric(
        "🔮 Embedded Prediction",
        f"{predicted_temperature:.2f} °C"
    )

with c:

    st.metric(
        "📈 Thermal Stress",
        f"{thermal_stress:.1f}"
    )


a, b, c = st.columns(3)

with a:

    st.metric(
        "❄️ Cooling",
        "ON" if cooling_active else "OFF"
    )

with b:

    st.metric(
        "🔥 Hotspot",
        hotspot_status
    )

with c:

    st.metric(
        "🛡️ Protection",
        protection_status
    )


# =========================================================
# STATUS MESSAGE
# =========================================================

if temperature >= 50:

    st.error(
        "🔴 CRITICAL: Server temperature has reached the critical range."
    )

elif temperature >= 40:

    st.warning(
        "🟠 HIGH TEMPERATURE: Server requires active monitoring."
    )

elif temperature >= 35:

    st.warning(
        "🟡 WARM: Temperature is above the normal operating range."
    )

else:

    st.success(
        "🟢 SAFE: Server temperature is within the safe range."
    )


# =========================================================
# ML PREDICTIONS
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">🤖 AI / ML Predictions</div>',
    unsafe_allow_html=True
)

ml = run_ml_predictions(
    server
)


a, b = st.columns(2)

with a:

    st.markdown("#### Model 1 — Future Temperature")

    st.metric(
        "Predicted Temperature",
        f"{ml['future_temperature']:.2f} °C"
    )

    st.caption(
        "Predicted future thermal condition."
    )

with b:

    st.markdown("#### Model 2 — Required Fan Speed")

    st.metric(
        "Recommended Fan Speed",
        f"{ml['fan_speed']:.1f}%"
    )

    st.caption(
        "AI-estimated cooling requirement."
    )


a, b = st.columns(2)

with a:

    st.markdown("#### Model 3 — Hotspot Risk")

    st.metric(
        "Hotspot Risk",
        f"{ml['hotspot_risk']:.1f}%"
    )

    if ml["hotspot_risk"] >= 70:

        st.error("High hotspot risk")

    elif ml["hotspot_risk"] >= 40:

        st.warning("Moderate hotspot risk")

    else:

        st.success("Low hotspot risk")

with b:

    st.markdown("#### Model 4 — Cooling Effectiveness")

    st.metric(
        "Cooling Effectiveness",
        f"{ml['cooling_effectiveness']:.1f}%"
    )


# =========================================================
# MODEL 5
# =========================================================

st.markdown("#### Model 5 — Overheating Warning")

if ml["overheating_warning"] == 1:

    st.error(
        "🚨 OVERHEATING WARNING"
    )

else:

    st.success(
        "✅ No overheating warning"
    )


# =========================================================
# SYSTEM EXPLANATION
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">🧠 System Interpretation</div>',
    unsafe_allow_html=True
)

if temperature >= 50:

    explanation = (
        f"Server {server_id} is currently at "
        f"{temperature:.2f} °C. The system considers "
        f"this a critical thermal condition. "
        f"Cooling is {'active' if cooling_active else 'not active'} "
        f"and the protection status is {protection_status}."
    )

elif temperature >= 40:

    explanation = (
        f"Server {server_id} is currently at "
        f"{temperature:.2f} °C. The system has detected "
        f"elevated temperature and recommends monitoring "
        f"the cooling response."
    )

else:

    explanation = (
        f"Server {server_id} is operating at "
        f"{temperature:.2f} °C. The current thermal condition "
        f"is within the normal operating range."
    )

st.info(
    explanation
)


# =========================================================
# RAW MQTT DATA
# =========================================================

with st.expander("🔧 View Live MQTT Data"):

    st.json(
        live_data
    )


# =========================================================
# AUTO REFRESH EVERY 3 SECONDS
# =========================================================

@st.fragment(run_every=3)
def refresh_status():

    if (
        st.session_state.get(
            "user_connected",
            False
        )
    ):

        st.caption(
            "🔄 Dashboard automatically checks for new MQTT data every 3 seconds."
        )


refresh_status()
