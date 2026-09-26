import streamlit as st

st.set_page_config(
    page_title="AI Server Hotspot Cooling",
    page_icon="❄️",
    layout="wide"
)

st.title("❄️ AI Server Hotspot Cooling System")
st.caption("AI-powered server temperature monitoring and cooling")

st.sidebar.header("Device Connection")

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

if "connected" not in st.session_state:
    st.session_state.connected = False

if st.sidebar.button(
    "Disconnect" if st.session_state.connected else "Connect",
    use_container_width=True
):
    st.session_state.connected = not st.session_state.connected

if st.session_state.connected:

    st.success(f"Connected to {device_id}")

    st.subheader("🖥️ Connected Servers")

    columns = st.columns(4)

    for i in range(server_count):
        with columns[i % 4]:
            st.info(f"Server {i + 1}")
            st.metric("Temperature", "Waiting...")
            st.write("Status: Waiting for data")

    st.divider()

    st.subheader("📊 Server Details")

    selected_server = st.selectbox(
        "Select Server",
        [f"Server {i + 1}" for i in range(server_count)]
    )

    st.write(f"### {selected_server}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Current Temperature", "-- °C")
        st.metric("AI Predicted Temperature", "-- °C")

    with col2:
        st.metric("Required Fan Speed", "-- %")
        st.metric("Hotspot Risk", "-- %")

    with col3:
        st.metric("Cooling Effectiveness", "-- %")
        st.metric("Thermal Stress", "--")

    st.write("**Hardware Status:** Waiting for Wokwi data")
    st.write("**Cooling:** Waiting for Wokwi data")
    st.write("**Hotspot:** Waiting for AI data")
    st.write("**Protection:** Waiting for Wokwi data")

else:

    st.warning("🔌 Device disconnected")

    st.info(
        "Connect your ESP32/Wokwi device to start monitoring servers."
    )
