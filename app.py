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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Server Hotspot Cooling System",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
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
    margin-bottom: 15px;
}

.safe {
    color: #5ee58a;
    font-weight: 700;
}

.warning {
    color: #ffd166;
    font-weight: 700;
}

.hotspot {
    color: #ff8c42;
    font-weight: 700;
}

.critical {
    color: #ff5c5c;
    font-weight: 700;
}

.live {
    color: #5ee58a;
    font-weight: 700;
}

.small-note {
    color: #9aa4b2;
    font-size: 0.9rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ZIP_FILE = BASE_DIR / "server_cooling_5_models_1000_rows.zip"

DATA_DIR = BASE_DIR / "_model_data"


# ============================================================
# MQTT CONFIGURATION
#
# WOKWI:
#   broker.hivemq.com
#   port 1883
#
# STREAMLIT:
#   broker.hivemq.com
#   WebSocket port 8000
#   path /mqtt
# ============================================================

MQTT_BROKER = "broker.hivemq.com"

MQTT_TCP_PORT = 1883

MQTT_WEBSOCKET_PORT = 8000

MQTT_TOPIC = "ai-cooling/WOKWI-ESP32-001"


# ============================================================
# TEMPERATURE THRESHOLDS
# ============================================================

SAFE_TEMP = 35.0

WARNING_TEMP = 40.0

CRITICAL_TEMP = 50.0


# ============================================================
# MODEL FEATURES
# ============================================================

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


# ============================================================
# CSV FILE NAMES INSIDE ZIP
# ============================================================

CSV_NAMES = [

    "server_cooling_model1_1000_rows.csv",

    "server_cooling_model2_1000_rows.csv",

    "server_cooling_model3_1000_rows.csv",

    "server_cooling_model4_1000_rows.csv",

    "server_cooling_model5_1000_rows.csv"

]


# ============================================================
# DATASET PREPARATION
# ============================================================

def prepare_datasets():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Check whether files were already extracted
    # --------------------------------------------------------

    found_files = {}

    for file in DATA_DIR.rglob("*.csv"):

        found_files[file.name] = file


    if all(
        name in found_files
        for name in CSV_NAMES
    ):

        return [
            found_files[name]
            for name in CSV_NAMES
        ]


    # --------------------------------------------------------
    # Check ZIP
    # --------------------------------------------------------

    if not ZIP_FILE.exists():

        raise FileNotFoundError(

            "\nDataset ZIP file was not found.\n\n"

            f"Expected:\n{ZIP_FILE}\n\n"

            "Your GitHub repository must contain:\n"

            "app.py\n"

            "requirements.txt\n"

            "server_cooling_5_models_1000_rows.zip"

        )


    # --------------------------------------------------------
    # Extract ZIP
    # --------------------------------------------------------

    try:

        with zipfile.ZipFile(
            ZIP_FILE,
            "r"
        ) as zip_ref:

            zip_ref.extractall(
                DATA_DIR
            )

    except zipfile.BadZipFile:

        raise FileNotFoundError(
            "server_cooling_5_models_1000_rows.zip "
            "is not a valid ZIP file."
        )


    # --------------------------------------------------------
    # Search recursively
    # --------------------------------------------------------

    found_files = {}

    for file in DATA_DIR.rglob("*.csv"):

        found_files[file.name] = file


    missing = [

        name

        for name in CSV_NAMES

        if name not in found_files

    ]


    if missing:

        raise FileNotFoundError(

            "These CSV files were not found inside the ZIP:\n\n"

            + "\n".join(missing)

        )


    return [

        found_files[name]

        for name in CSV_NAMES

    ]


# ============================================================
# LOAD DATASETS
# ============================================================

@st.cache_data
def load_datasets():
    files = prepare_datasets()

    if len(files) != 5:
        raise RuntimeError(
            f"Expected 5 CSV files, but found {len(files)}"
        )

    datasets = []

    for file in files:
        df = pd.read_csv(file)

        # Clean column names
        df.columns = (
            df.columns
            .astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
        )

        # Remove accidental unnamed/index columns
        drop_cols = [
            col for col in df.columns
            if col.lower().startswith("unnamed")
        ]

        if drop_cols:
            df = df.drop(columns=drop_cols)

        datasets.append(df)

    return datasets

# ============================================================
# MQTT STATE
# ============================================================

class MQTTState:

    def __init__(self):

        self.lock = threading.Lock()

        self.connected = False

        self.latest_data = None

        self.last_update = None

        self.message_count = 0

        self.connection_error = None

        client_id = (
            "STREAMLIT-"
            + uuid.uuid4().hex[:12]
        )

        # ----------------------------------------------------
        # MQTT OVER WEBSOCKETS
        # This is important for Streamlit Cloud.
        # ----------------------------------------------------

        self.client = mqtt.Client(

            callback_api_version=
            mqtt.CallbackAPIVersion.VERSION2,

            client_id=client_id,

            transport="websockets"

        )

        self.client.ws_set_options(
            path="/mqtt"
        )

        self.client.on_connect = (
            self.on_connect
        )

        self.client.on_message = (
            self.on_message
        )

        self.client.on_disconnect = (
            self.on_disconnect
        )


    # ========================================================
    # CONNECT CALLBACK
    # ========================================================

    def on_connect(
        self,
        client,
        userdata,
        flags,
        reason_code,
        properties
    ):

        print(
            "MQTT connection result:",
            reason_code
        )

        if reason_code == 0:

            with self.lock:

                self.connected = True

                self.connection_error = None


            result = client.subscribe(
                MQTT_TOPIC,
                qos=0
            )

            print(
                "Subscribed:",
                MQTT_TOPIC,
                result
            )

        else:

            with self.lock:

                self.connected = False

                self.connection_error = (
                    f"MQTT connection refused: "
                    f"{reason_code}"
                )


    # ========================================================
    # MESSAGE CALLBACK
    # ========================================================

    def on_message(
        self,
        client,
        userdata,
        msg
    ):

        try:

            payload = msg.payload.decode(
                "utf-8"
            )

            print(
                "MQTT MESSAGE RECEIVED:"
            )

            print(payload)


            data = json.loads(
                payload
            )


            # -----------------------------------------------
            # Verify expected structure
            # -----------------------------------------------

            if "servers" not in data:

                print(
                    "MQTT message does not contain servers."
                )

                return


            # -----------------------------------------------
            # Store newest Wokwi data
            # -----------------------------------------------

            with self.lock:

                self.latest_data = data

                self.last_update = time.time()

                self.message_count += 1


        except Exception as e:

            print(
                "MQTT message processing error:",
                e
            )


    # ========================================================
    # DISCONNECT CALLBACK
    # ========================================================

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


        print(
            "MQTT disconnected:",
            reason_code
        )


    # ========================================================
    # CONNECT
    # ========================================================

    def connect(self):

        try:

            if self.client.is_connected():

                with self.lock:

                    self.connected = True

                return True


            # ------------------------------------------------
            # Streamlit Cloud connects using WebSocket 8000
            # ------------------------------------------------

            self.client.connect(

                MQTT_BROKER,

                MQTT_WEBSOCKET_PORT,

                keepalive=60

            )


            # ------------------------------------------------
            # Start continuous MQTT listener
            # ------------------------------------------------

            self.client.loop_start()


            return True


        except Exception as e:

            print(
                "MQTT connection error:",
                e
            )

            with self.lock:

                self.connected = False

                self.connection_error = str(e)


            return False


    # ========================================================
    # DISCONNECT
    # ========================================================

    def disconnect(self):

        try:

            self.client.loop_stop()

            self.client.disconnect()

        except Exception:

            pass


        with self.lock:

            self.connected = False


    # ========================================================
    # GET DATA
    # ========================================================

    def get_data(self):

        with self.lock:

            return self.latest_data


    # ========================================================
    # GET CONNECTION STATUS
    # ========================================================

    def is_connected(self):

        with self.lock:

            return self.connected


    # ========================================================
    # GET LAST UPDATE
    # ========================================================

    def get_last_update(self):

        with self.lock:

            return self.last_update


    # ========================================================
    # GET MESSAGE COUNT
    # ========================================================

    def get_message_count(self):

        with self.lock:

            return self.message_count


    # ========================================================
    # GET ERROR
    # ========================================================

    def get_error(self):

        with self.lock:

            return self.connection_error


# ============================================================
# PERSISTENT MQTT OBJECT
# ============================================================

@st.cache_resource
def get_mqtt_state():

    return MQTTState()


mqtt_state = get_mqtt_state()


# ============================================================
# TRAIN ML MODELS
# ============================================================

@st.cache_resource
def train_models():

    df1, df2, df3, df4, df5 = load_datasets()

    # -------------------------------------------------
    # Check and clean every dataset
    # -------------------------------------------------

    datasets = [df1, df2, df3, df4, df5]

    targets = [
        "future_GPU_temperature_C",
        "required_fan_speed_percent",
        "hotspot_risk_percent",
        "cooling_effectiveness_percent",
        "overheating_warning",
    ]

    for i, (df, target) in enumerate(
        zip(datasets, targets), start=1
    ):

        # Normalize column names again
        df.columns = (
            df.columns
            .astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
        )

        # Convert all expected feature columns to numeric
        for feature in FEATURES:
            if feature in df.columns:
                df[feature] = pd.to_numeric(
                    df[feature],
                    errors="coerce"
                )

        # Convert target to numeric
        if target in df.columns:
            df[target] = pd.to_numeric(
                df[target],
                errors="coerce"
            )

        # Check features
        missing_features = [
            feature
            for feature in FEATURES
            if feature not in df.columns
        ]

        if missing_features:
            st.error(
                f"Model {i} CSV does not contain the required columns."
            )

            st.write(
                "Required feature columns:"
            )
            st.write(FEATURES)

            st.write(
                "Columns actually found in this CSV:"
            )
            st.write(list(df.columns))

            st.stop()

        # Check target
        if target not in df.columns:
            st.error(
                f"Model {i} is missing target column: {target}"
            )

            st.write(
                "Columns actually found:"
            )
            st.write(list(df.columns))

            st.stop()

        # Remove invalid rows
        df.dropna(
            subset=FEATURES + [target],
            inplace=True
        )

    # -------------------------------------------------
    # MODEL 1
    # -------------------------------------------------

    X1 = df1[FEATURES]
    y1 = df1["future_GPU_temperature_C"]

    model1 = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    model1.fit(X1, y1)

    # -------------------------------------------------
    # MODEL 2
    # -------------------------------------------------

    X2 = df2[FEATURES]
    y2 = df2["required_fan_speed_percent"]

    model2 = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    model2.fit(X2, y2)

    # -------------------------------------------------
    # MODEL 3
    # -------------------------------------------------

    X3 = df3[FEATURES]
    y3 = df3["hotspot_risk_percent"]

    model3 = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    model3.fit(X3, y3)

    # -------------------------------------------------
    # MODEL 4
    # -------------------------------------------------

    X4 = df4[FEATURES]
    y4 = df4["cooling_effectiveness_percent"]

    model4 = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    model4.fit(X4, y4)

    # -------------------------------------------------
    # MODEL 5
    # -------------------------------------------------

    X5 = df5[FEATURES]
    y5 = df5["overheating_warning"]

    model5 = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    model5.fit(X5, y5)

    return {
        "model1": model1,
        "model2": model2,
        "model3": model3,
        "model4": model4,
        "model5": model5,
    }
# ============================================================
# TRAIN
# ============================================================

models = train_models()


# ============================================================
# BUILD ML INPUTS FROM LIVE WOKWI TEMPERATURE
# ============================================================

def build_features(server):

    temp = float(
        server.get(
            "temperature",
            22.0
        )
    )


    cooling = bool(
        server.get(
            "cooling_active",
            False
        )
    )


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

        "GPU_usage_percent":
        gpu_usage,

        "GPU_power_W":
        gpu_power,

        "CPU_usage_percent":
        cpu_usage,

        "CPU_temperature_C":
        cpu_temp,

        "GPU_temperature_C":
        temp,

        "ambient_temperature_C":
        ambient,

        "inlet_temperature_C":
        inlet,

        "outlet_temperature_C":
        outlet,

        "fan_speed_percent":
        fan_speed,

        "airflow_m3_s":
        airflow,

        "workload_percent":
        workload

    }])


# ============================================================
# ML PREDICTION
# ============================================================

def predict_server(server):

    features = build_features(
        server
    )


    future_temp = float(
        models["model1"].predict(
            features
        )[0]
    )


    required_fan = float(
        models["model2"].predict(
            features
        )[0]
    )


    hotspot_risk = float(
        models["model3"].predict(
            features
        )[0]
    )


    cooling_effectiveness = float(
        models["model4"].predict(
            features
        )[0]
    )


    warning = int(
        models["model5"].predict(
            features
        )[0]
    )


    temp = float(
        server.get(
            "temperature",
            0
        )
    )


    # --------------------------------------------------------
    # LIVE HARDWARE TEMPERATURE OVERRIDES
    # --------------------------------------------------------

    if temp >= CRITICAL_TEMP:

        hotspot_risk = 100.0

        warning = 1


    elif temp >= WARNING_TEMP:

        hotspot_risk = max(
            hotspot_risk,
            80.0
        )

        warning = 1


    elif temp >= SAFE_TEMP:

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


    required_fan = max(
        0.0,
        min(
            100.0,
            required_fan
        )
    )


    cooling_effectiveness = max(
        0.0,
        min(
            100.0,
            cooling_effectiveness
        )
    )


    return {

        "future_temp":
        future_temp,

        "required_fan":
        required_fan,

        "hotspot_risk":
        hotspot_risk,

        "cooling_effectiveness":
        cooling_effectiveness,

        "warning":
        warning

    }


# ============================================================
# STATUS FUNCTIONS
# ============================================================

def temperature_status(temp):

    if temp >= CRITICAL_TEMP:

        return "CRITICAL"


    if temp >= WARNING_TEMP:

        return "HIGH TEMPERATURE"


    if temp >= SAFE_TEMP:

        return "WARM - MONITORING"


    return "SAFE"


def protection_status(temp):

    if temp >= CRITICAL_TEMP:

        return "CRITICAL PROTECTION"


    if temp >= WARNING_TEMP:

        return "ACTIVE"


    return "NORMAL"


def hotspot_status(server):

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


    if temp >= CRITICAL_TEMP:

        return "CRITICAL HOTSPOT"


    if temp >= WARNING_TEMP:

        return "HOTSPOT / HIGH THERMAL STRESS"


    if predicted >= SAFE_TEMP:

        return "HOTSPOT PREDICTED"


    if cooling:

        return "COOLING ACTIVE"


    return "NO HOTSPOT"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ⚙️ System Control"
    )


    st.markdown(
        "### MQTT Device"
    )


    st.code(
        "WOKWI-ESP32-001"
    )


    st.write(
        "Broker:",
        MQTT_BROKER
    )


    st.write(
        "Wokwi MQTT Port:",
        MQTT_TCP_PORT
    )


    st.write(
        "Dashboard WebSocket Port:",
        MQTT_WEBSOCKET_PORT
    )


    st.write(
        "Topic:",
        MQTT_TOPIC
    )


    st.divider()


    # --------------------------------------------------------
    # CONNECT
    # --------------------------------------------------------

    if not mqtt_state.is_connected():

        if st.button(
            "🔌 CONNECT",
            use_container_width=True
        ):

            success = mqtt_state.connect()


            if success:

                st.success(
                    "Dashboard MQTT connection started."
                )

                st.rerun()

            else:

                st.error(
                    "Could not connect to MQTT."
                )


    else:

        if st.button(
            "🔴 DISCONNECT",
            use_container_width=True
        ):

            mqtt_state.disconnect()

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


    st.write(
        "Servers: 4"
    )


    st.write(
        "Dashboard refresh: 30 seconds"
    )


    st.write(
        "MQTT listener: continuous"
    )


    st.divider()


    count = mqtt_state.get_message_count()


    st.metric(
        "MQTT Messages Received",
        count
    )


    error = mqtt_state.get_error()


    if error:

        st.error(
            error
        )


    st.divider()


    st.caption(
        "AI Server Hotspot Cooling System"
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.title(
    "🌡️ AI Server Hotspot Cooling System"
)


st.write(
    "Real-time Wokwi server temperature monitoring, "
    "hotspot detection and AI cooling analysis."
)


# ============================================================
# LIVE DASHBOARD
#
# IMPORTANT:
# The MQTT listener runs continuously.
#
# The Streamlit UI refreshes every 30 seconds.
# ============================================================

@st.fragment(run_every=30)
def realtime_dashboard():

    # --------------------------------------------------------
    # Get newest MQTT message
    # --------------------------------------------------------

    data = mqtt_state.get_data()


    if data is None:

        st.info(
            "Waiting for data from Wokwi..."
        )


        st.caption(
            "Click CONNECT and keep the Wokwi simulation running."
        )


        return


    # --------------------------------------------------------
    # Servers
    # --------------------------------------------------------

    servers = data.get(
        "servers",
        []
    )


    if not servers:

        st.warning(
            "MQTT message received, but no servers were found."
        )


        return


    # --------------------------------------------------------
    # LAST MESSAGE TIME
    # --------------------------------------------------------

    last_update = mqtt_state.get_last_update()


    if last_update is not None:

        age = (
            time.time()
            - last_update
        )


        if age <= 35:

            st.markdown(
                '<span class="live">'
                '🟢 LIVE — Latest Wokwi data received'
                '</span>',
                unsafe_allow_html=True
            )


        else:

            st.warning(
                f"Last MQTT message was received "
                f"{age:.0f} seconds ago."
            )


    # --------------------------------------------------------
    # TEMPERATURES
    # --------------------------------------------------------

    temperatures = [

        float(
            server.get(
                "temperature",
                0
            )
        )

        for server in servers

    ]


    average_temp = (
        sum(temperatures)
        /
        len(temperatures)
    )


    highest_temp = max(
        temperatures
    )


    hotspot_count = sum(

        temp >= WARNING_TEMP

        for temp in temperatures

    )


    critical_count = sum(

        temp >= CRITICAL_TEMP

        for temp in temperatures

    )


    cooling_count = sum(

        bool(
            server.get(
                "cooling_active",
                False
            )
        )

        for server in servers

    )


    # ========================================================
    # TOP KPIs
    # ========================================================

    st.markdown(
        "## 📡 Live Wokwi Data"
    )


    m1, m2, m3, m4, m5 = st.columns(5)


    m1.metric(
        "Servers",
        len(servers)
    )


    m2.metric(
        "Average Temperature",
        f"{average_temp:.1f} °C"
    )


    m3.metric(
        "Highest Temperature",
        f"{highest_temp:.1f} °C"
    )


    m4.metric(
        "Hotspots",
        hotspot_count
    )


    m5.metric(
        "Cooling Active",
        cooling_count
    )


    if critical_count > 0:

        st.error(
            f"🚨 {critical_count} server(s) "
            f"are at critical temperature."
        )


    # ========================================================
    # SERVER CARDS
    # ========================================================

    st.markdown(
        "## 🖥️ Live Server Monitoring"
    )


    columns = st.columns(2)


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


        status = temperature_status(
            temperature
        )


        with columns[index % 2]:

            st.markdown(
                '<div class="server-card">',
                unsafe_allow_html=True
            )


            st.markdown(
                f"### 🖥️ Server {server_id}"
            )


            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            if temperature >= CRITICAL_TEMP:

                st.markdown(
                    '<span class="critical">'
                    'CRITICAL'
                    '</span>',
                    unsafe_allow_html=True
                )


            elif temperature >= WARNING_TEMP:

                st.markdown(
                    '<span class="hotspot">'
                    'HIGH TEMPERATURE'
                    '</span>',
                    unsafe_allow_html=True
                )


            elif temperature >= SAFE_TEMP:

                st.markdown(
                    '<span class="warning">'
                    'WARM - MONITORING'
                    '</span>',
                    unsafe_allow_html=True
                )


            else:

                st.markdown(
                    '<span class="safe">'
                    'SAFE'
                    '</span>',
                    unsafe_allow_html=True
                )


            st.write("")


            a, b = st.columns(2)


            a.metric(
                "Wokwi Temperature",
                f"{temperature:.2f} °C"
            )


            b.metric(
                "Wokwi Prediction",
                f"{predicted_temperature:.2f} °C"
            )


            c, d = st.columns(2)


            c.metric(
                "Cooling",
                "ON" if cooling else "OFF"
            )


            d.metric(
                "Thermal Stress",
                f"{stress:.1f}"
            )


            # ------------------------------------------------
            # HOTSPOT
            # ------------------------------------------------

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
                f"**Hardware Status:** {status}"
            )


            st.write(
                f"**Protection:** "
                f"{protection_status(temperature)}"
            )


            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


    # ========================================================
    # RAW MQTT DATA
    # ========================================================

    with st.expander(
        "📨 View Raw Wokwi MQTT Data"
    ):

        st.json(
            data
        )


    # ========================================================
    # HARDWARE TABLE
    # ========================================================

    st.markdown(
        "## 🔧 Real-Time Hardware Status"
    )


    rows = []


    for server in servers:

        sid = server.get(
            "server_id",
            0
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


        rows.append({

            "Server":
            f"Server {sid}",

            "Wokwi Temperature (°C)":
            round(
                temp,
                2
            ),

            "Predicted Temperature (°C)":
            round(
                predicted,
                2
            ),

            "Status":
            temperature_status(
                temp
            ),

            "Hotspot":
            hotspot_status(
                server
            ),

            "Cooling":
            "ON"
            if cooling
            else
            "OFF",

            "Protection":
            protection_status(
                temp
            )

        })


    st.dataframe(

        pd.DataFrame(
            rows
        ),

        use_container_width=True,

        hide_index=True

    )


    # ========================================================
    # SERVER DETAIL
    # ========================================================

    st.markdown(
        "## 🔍 Server Detail Analysis"
    )


    server_ids = [

        int(
            server.get(
                "server_id",
                i + 1
            )
        )

        for i, server
        in enumerate(servers)

    ]


    selected_server_id = st.selectbox(

        "Select Server",

        server_ids

    )


    selected_server = None


    for server in servers:

        if int(

            server.get(
                "server_id",
                0
            )

        ) == selected_server_id:

            selected_server = server

            break


    if selected_server is not None:

        result = predict_server(
            selected_server
        )


        features = build_features(
            selected_server
        ).iloc[0]


        st.markdown(
            f"### Server {selected_server_id}"
        )


        x1, x2, x3, x4 = st.columns(4)


        x1.metric(
            "Current Wokwi Temperature",
            f'{selected_server.get("temperature", 0):.2f} °C'
        )


        x2.metric(
            "AI Future GPU Temperature",
            f'{result["future_temp"]:.2f} °C'
        )


        x3.metric(
            "AI Required Fan Speed",
            f'{result["required_fan"]:.1f}%'
        )


        x4.metric(
            "AI Hotspot Risk",
            f'{result["hotspot_risk"]:.1f}%'
        )


        # ----------------------------------------------------
        # ML INPUTS
        # ----------------------------------------------------

        st.markdown(
            "### 📊 ML Input Parameters"
        )


        st.caption(
            "Temperature comes directly from Wokwi. "
            "The remaining ML inputs are estimated from the "
            "live temperature for this demonstration."
        )


        p1, p2, p3, p4 = st.columns(4)


        p1.metric(
            "GPU Usage",
            f'{features["GPU_usage_percent"]:.1f}%'
        )


        p2.metric(
            "GPU Power",
            f'{features["GPU_power_W"]:.1f} W'
        )


        p3.metric(
            "CPU Usage",
            f'{features["CPU_usage_percent"]:.1f}%'
        )


        p4.metric(
            "CPU Temperature",
            f'{features["CPU_temperature_C"]:.1f} °C'
        )


        p5, p6, p7, p8 = st.columns(4)


        p5.metric(
            "Ambient",
            f'{features["ambient_temperature_C"]:.1f} °C'
        )


        p6.metric(
            "Inlet",
            f'{features["inlet_temperature_C"]:.1f} °C'
        )


        p7.metric(
            "Outlet",
            f'{features["outlet_temperature_C"]:.1f} °C'
        )


        p8.metric(
            "Airflow",
            f'{features["airflow_m3_s"]:.2f} m³/s'
        )


        p9, p10 = st.columns(2)


        p9.metric(
            "Workload",
            f'{features["workload_percent"]:.1f}%'
        )


        p10.metric(
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
            f'{result["future_temp"]:.2f} °C'
        )


        r2.metric(
            "Model 2",
            f'{result["required_fan"]:.1f}%'
        )


        r3.metric(
            "Model 3",
            f'{result["hotspot_risk"]:.1f}%'
        )


        r4.metric(
            "Model 4",
            f'{result["cooling_effectiveness"]:.1f}%'
        )


        r5.metric(
            "Model 5",
            "WARNING"
            if result["warning"]
            else "NORMAL"
        )


        # ----------------------------------------------------
        # AI EXPLANATION
        # ----------------------------------------------------

        st.markdown(
            "### 🧠 AI Explanation"
        )


        current_temp = float(
            selected_server.get(
                "temperature",
                0
            )
        )


        if current_temp >= CRITICAL_TEMP:

            st.error(
                "The Wokwi server temperature is in the "
                "critical range. Cooling protection should "
                "remain active."
            )


        elif current_temp >= WARNING_TEMP:

            st.warning(
                "The Wokwi server temperature is in the "
                "high-temperature range. The system identifies "
                "this as a hotspot/high thermal-stress condition."
            )


        elif result["future_temp"] >= SAFE_TEMP:

            st.info(
                "The AI model predicts that the server may "
                "approach the hotspot threshold."
            )


        else:

            st.success(
                "The current Wokwi temperature is within "
                "the monitored safe range."
            )


    # ========================================================
    # ANALYTICS
    # ========================================================

    st.markdown(
        "## 📈 Real-Time Analytics"
    )


    chart_rows = []


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


        result = predict_server(
            server
        )


        chart_rows.append({

            "Server":
            f"Server {sid}",

            "Temperature":
            temp,

            "Risk":
            result["hotspot_risk"],

            "Cooling":
            result[
                "cooling_effectiveness"
            ]

        })


    chart_df = pd.DataFrame(
        chart_rows
    )


    # --------------------------------------------------------
    # Temperature chart
    # --------------------------------------------------------

    fig1 = go.Figure()


    fig1.add_trace(

        go.Bar(

            x=chart_df["Server"],

            y=chart_df["Temperature"],

            name="Temperature"

        )

    )


    fig1.add_hline(

        y=SAFE_TEMP,

        line_dash="dash",

        annotation_text="Safe"

    )


    fig1.add_hline(

        y=WARNING_TEMP,

        line_dash="dash",

        annotation_text="Warning"

    )


    fig1.add_hline(

        y=CRITICAL_TEMP,

        line_dash="dash",

        annotation_text="Critical"

    )


    fig1.update_layout(

        title="Current Wokwi Server Temperatures",

        template="plotly_dark",

        height=400,

        xaxis_title="Server",

        yaxis_title="Temperature (°C)"

    )


    st.plotly_chart(

        fig1,

        use_container_width=True

    )


    # --------------------------------------------------------
    # Risk chart
    # --------------------------------------------------------

    fig2 = go.Figure()


    fig2.add_trace(

        go.Bar(

            x=chart_df["Server"],

            y=chart_df["Risk"],

            name="Hotspot Risk"

        )

    )


    fig2.update_layout(

        title="AI Hotspot Risk",

        template="plotly_dark",

        height=400,

        xaxis_title="Server",

        yaxis_title="Risk (%)",

        yaxis_range=[
            0,
            100
        ]

    )


    st.plotly_chart(

        fig2,

        use_container_width=True

    )


    # --------------------------------------------------------
    # Cooling effectiveness
    # --------------------------------------------------------

    fig3 = go.Figure()


    fig3.add_trace(

        go.Bar(

            x=chart_df["Server"],

            y=chart_df["Cooling"],

            name="Cooling Effectiveness"

        )

    )


    fig3.update_layout(

        title="AI Cooling Effectiveness",

        template="plotly_dark",

        height=400,

        xaxis_title="Server",

        yaxis_title="Effectiveness (%)",

        yaxis_range=[
            0,
            100
        ]

    )


    st.plotly_chart(

        fig3,

        use_container_width=True

    )


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.markdown(
        "## 📊 Model Performance"
    )


    performance = pd.DataFrame([

        {

            "Model":
            "Model 1 - Future GPU Temperature",

            "R²":
            models[
                "model1_metrics"
            ]["R2"],

            "MAE":
            models[
                "model1_metrics"
            ]["MAE"],

            "RMSE":
            models[
                "model1_metrics"
            ]["RMSE"]

        },

        {

            "Model":
            "Model 2 - Required Fan Speed",

            "R²":
            models[
                "model2_metrics"
            ]["R2"],

            "MAE":
            models[
                "model2_metrics"
            ]["MAE"],

            "RMSE":
            models[
                "model2_metrics"
            ]["RMSE"]

        },

        {

            "Model":
            "Model 3 - Hotspot Risk",

            "R²":
            models[
                "model3_metrics"
            ]["R2"],

            "MAE":
            models[
                "model3_metrics"
            ]["MAE"],

            "RMSE":
            models[
                "model3_metrics"
            ]["RMSE"]

        },

        {

            "Model":
            "Model 4 - Cooling Effectiveness",

            "R²":
            models[
                "model4_metrics"
            ]["R2"],

            "MAE":
            models[
                "model4_metrics"
            ]["MAE"],

            "RMSE":
            models[
                "model4_metrics"
            ]["RMSE"]

        }

    ])


    st.dataframe(

        performance,

        use_container_width=True,

        hide_index=True

    )


# ============================================================
# START DASHBOARD
# ============================================================

realtime_dashboard()

