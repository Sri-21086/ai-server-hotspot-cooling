import streamlit as st
import paho.mqtt.client as mqtt
import json
import threading
import time

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AI Server Hotspot Cooling",
    page_icon="❄️",
    layout="wide"
)

# ============================================================
# MQTT SETTINGS
# ============================================================

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "ai-cooling/WOKWI-ESP32-001"


# ============================================================
# GLOBAL MQTT DATA
# ============================================================

@st.cache_resource
def get_mqtt_system():

    system = {
        "client": None,
        "data": {},
        "connected": False,
        "last_update": None,
        "lock": threading.Lock()
    }

    def on_connect(client, userdata, flags, reason_code, properties=None):

        if reason_code == 0:

            system["connected"] = True

            client.subscribe(MQTT_TOPIC)

        else:

            system["connected"] = False

    def on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):

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

            print("MQTT message error:", e)

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="streamlit-ai-cooling-dashboard"
    )

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    system["client"] = client

    return system


mqtt_system = get_mqtt_system()


# ============================================================
# MQTT CONNECT
# ============================================================

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


# ============================================================
# MQTT DISCONNECT
# ============================================================

def disconnect_mqtt():

    try:

        if mqtt_system["connected"]:

            mqtt_system["client"].loop_stop()

            mqtt_system["client"].disconnect()

            mqtt_system["connected"] = False

    except Exception:

        pass


# ============================================================
# TITLE
# ============================================================

st.title("❄️ AI Server Hotspot Cooling System")

st.caption(
    "Continuous server monitoring • AI prediction • hotspot detection • cooling protection"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🔌 Device Connection")

device_id = st.sidebar.text_input(
    "Device ID",
    value="WOKWI-ESP32-001"
)

server_count = st.sidebar.number_input(
    "Number of Servers",
    min_value=1,
    max_value=20,
    value=4,
    step=1
)


# ============================================================
# CONNECTION BUTTON
# ============================================================

if "user_connected" not in st.session_state:

    st.session_state.user_connected = False


if st.sidebar.button(
    "Disconnect" if st.session_state.user_connected else "Connect",
    use_container_width=True
):

    if not st.session_state.user_connected:

        if connect_mqtt():

            st.session_state.user_connected = True

    else:

        disconnect_mqtt()

        st.session_state.user_connected = False


# ============================================================
# CONNECTION STATUS
# ============================================================

if st.session_state.user_connected and mqtt_system["connected"]:

    st.sidebar.success("🟢 MQTT Connected")

else:

    st.sidebar.error("🔴 Device Disconnected")


# ============================================================
# DISCONNECTED SCREEN
# ============================================================

if not st.session_state.user_connected:

    st.warning("🔌 Device is disconnected.")

    st.info(
        "Click Connect to start receiving live Wokwi server data."
    )

    st.stop()


# ============================================================
# READ MQTT DATA
# ============================================================

with mqtt_system["lock"]:

    live_data = dict(mqtt_system["data"])

    last_update = mqtt_system["last_update"]


# ============================================================
# WAITING FOR WOKWI
# ============================================================

if not live_data:

    st.warning(
        "⏳ Connected, but waiting for Wokwi data..."
    )

    st.info(
        "Start the Wokwi simulation and make sure MQTT DATA SENT appears in Serial Monitor."
    )

    time.sleep(1)

    st.rerun()


# ============================================================
# DEVICE INFORMATION
# ============================================================

received_device = live_data.get(
    "device_id",
    device_id
)

servers = live_data.get(
    "servers",
    []
)

st.success(
    f"🟢 Live device connected: {received_device}"
)


# ============================================================
# LAST UPDATE
# ============================================================

if last_update:

    seconds = time.time() - last_update

    if seconds < 10:

        st.caption(
            f"🟢 Live data received {seconds:.1f} seconds ago"
        )

    else:

        st.warning(
            f"⚠️ No new Wokwi data for {seconds:.1f} seconds"
        )


# ============================================================
# SERVER COUNT
# ============================================================

st.subheader("🖥️ Connected Servers")

actual_server_count = min(
    int(server_count),
    len(servers)
)

if actual_server_count == 0:

    st.warning("No server data received.")

else:

    cols = st.columns(4)

    for i in range(actual_server_count):

        server = servers[i]

        server_id = server.get(
            "server_id",
            i + 1
        )

        temperature = server.get(
            "temperature",
            0
        )

        status = server.get(
            "status",
            "UNKNOWN"
        )

        cooling = server.get(
            "cooling_active",
            False
        )

        with cols[i % 4]:

            st.markdown(
                f"### 🖥️ Server {server_id}"
            )

            st.metric(
                "Current Temperature",
                f"{temperature:.2f} °C"
            )

            if status == "SAFE":

                st.success("🟢 SAFE")

            elif status == "CRITICAL":

                st.error("🔴 CRITICAL")

            else:

                st.warning(f"🟠 {status}")

            if cooling:

                st.info("❄️ Cooling ON")

            else:

                st.write("Cooling OFF")


# ============================================================
# SERVER SELECTION
# ============================================================

st.divider()

st.subheader("🔎 Detailed Server Monitoring")

server_options = [
    f"Server {s.get('server_id', i + 1)}"
    for i, s in enumerate(servers[:actual_server_count])
]

selected_server = st.selectbox(
    "Select a server",
    server_options
)

selected_index = server_options.index(
    selected_server
)

server = servers[selected_index]


# ============================================================
# WOKWI OUTPUTS
# ============================================================

server_id = server.get(
    "server_id",
    selected_index + 1
)

temperature = server.get(
    "temperature",
    0
)

predicted_temperature = server.get(
    "predicted_temperature",
    0
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

thermal_stress = server.get(
    "thermal_stress",
    0
)

protection_status = server.get(
    "protection_status",
    "UNKNOWN"
)


# ============================================================
# MAIN SERVER INFORMATION
# ============================================================

st.markdown(
    f"## 🖥️ Server {server_id}"
)

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "🌡️ Current Temperature",
        f"{temperature:.2f} °C"
    )

    st.metric(
        "🔮 Predicted Temperature",
        f"{predicted_temperature:.2f} °C"
    )


with col2:

    st.metric(
        "📈 Thermal Stress",
        f"{thermal_stress:.1f}"
    )

    st.metric(
        "❄️ Cooling",
        "ON" if cooling_active else "OFF"
    )


with col3:

    st.metric(
        "🔥 Hotspot",
        hotspot_status
    )

    st.metric(
        "🛡️ Protection",
        protection_status
    )


# ============================================================
# HARDWARE STATUS
# ============================================================

st.subheader("🖥️ Hardware Status")

if status == "SAFE":

    st.success(
        "🟢 SAFE — Server temperature is within safe range."
    )

elif status == "CRITICAL":

    st.error(
        "🔴 CRITICAL — High thermal stress detected."
    )

else:

    st.warning(
        f"🟠 {status}"
    )


# ============================================================
# COOLING STATUS
# ============================================================

st.subheader("❄️ Cooling System")

if cooling_active:

    st.error(
        "❄️ COOLING ACTIVE — Protection response is ON."
    )

else:

    st.success(
        "Cooling is currently OFF."
    )


# ============================================================
# AI / ML OUTPUTS
# ============================================================

st.divider()

st.subheader("🤖 AI / ML Predictions")


ml1, ml2 = st.columns(2)


with ml1:

    st.markdown("### Model 1 — Future Temperature")

    st.metric(
        "Future Server/GPU Temperature",
        f"{predicted_temperature:.2f} °C"
    )

    st.caption(
        "Connected to the current embedded prediction; AutoTrain Model 1 will be connected next."
    )


with ml2:

    st.markdown("### Model 2 — Required Fan Speed")

    st.metric(
        "Required Fan Speed",
        "Waiting for Model 2"
    )

    st.caption(
        "AutoTrain Model 2 output will be connected here."
    )


ml3, ml4 = st.columns(2)


with ml3:

    st.markdown("### Model 3 — Hotspot Risk")

    if cooling_active:

        st.metric(
            "Hotspot Risk",
            "HIGH"
        )

    else:

        st.metric(
            "Hotspot Risk",
            "LOW"
        )

    st.caption(
        "AutoTrain Model 3 output will replace this fallback indicator."
    )


with ml4:

    st.markdown("### Model 4 — Cooling Effectiveness")

    if cooling_active:

        st.metric(
            "Cooling Effectiveness",
            "ACTIVE"
        )

    else:

        st.metric(
            "Cooling Effectiveness",
            "STANDBY"
        )

    st.caption(
        "AutoTrain Model 4 output will be connected here."
    )


# ============================================================
# MODEL 5
# ============================================================

st.subheader("⚠️ Overheating Warning")

if temperature >= 50:

    st.error(
        "🚨 OVERHEATING WARNING"
    )

elif temperature >= 40:

    st.warning(
        "⚠️ HIGH TEMPERATURE WARNING"
    )

else:

    st.success(
        "✅ No overheating warning"
    )

st.caption(
    "Current warning uses the Wokwi safety rule because Model 5 classification is not available in the current AutoTrain free setup."
)


# ============================================================
# AI EXPLANATION
# ============================================================

st.divider()

st.subheader("✨ AI Explanation")

if st.button(
    "✨ Explain Current Server Condition",
    use_container_width=True
):

    if temperature >= 50:

        explanation = (
            f"Server {server_id} is currently at "
            f"{temperature:.2f} °C, which is in the critical range. "
            f"The system has detected a hotspot and activated cooling. "
            f"Thermal stress is {thermal_stress:.1f} and protection status is "
            f"{protection_status}."
        )

    elif temperature >= 40:

        explanation = (
            f"Server {server_id} is at "
            f"{temperature:.2f} °C and requires monitoring. "
            f"Cooling status is "
            f"{'ON' if cooling_active else 'OFF'}."
        )

    else:

        explanation = (
            f"Server {server_id} is operating safely at "
            f"{temperature:.2f} °C. "
            f"No hotspot is currently detected and cooling is "
            f"{'active' if cooling_active else 'off'}."
        )

    st.info(explanation)


# ============================================================
# RAW DATA
# ============================================================

with st.expander("🔧 Live MQTT Data"):

    st.json(live_data)


# ============================================================
# AUTO REFRESH
# ============================================================

time.sleep(0.1)

st.rerun()
