
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="AI Server Hotspot Cooling System",
    page_icon="🖥️",
    layout="wide"
)

BASE = Path(_file_).resolve().parent

MODEL_FILES = {
    1: "server_cooling_model1_1000_rows.csv",
    2: "server_cooling_model2_1000_rows.csv",
    3: "server_cooling_model3_1000_rows.csv",
    4: "server_cooling_model4_1000_rows.csv",
    5: "server_cooling_model5_1000_rows.csv"
}

TARGETS = {
    1: "Future Temperature C",
    2: "Required Fan Speed Percent",
    3: "Hotspot Risk",
    4: "Cooling Effectiveness Percent",
    5: "Overheating Warning"
}


@st.cache_data
def load_data(model_number):
    file_path = BASE / MODEL_FILES[model_number]

    if not file_path.exists():
        return None

    return pd.read_csv(file_path)


def get_numeric_features(df, target):
    return [
        column
        for column in df.columns
        if column != target
        and pd.api.types.is_numeric_dtype(df[column])
    ]


def train_regression(df, target):
    features = get_numeric_features(df, target)

    data = df[features + [target]].dropna()

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return model, features, mae, rmse, r2


st.title("🖥️ AI-Based Server Hotspot Cooling System")

st.write(
    "AI-assisted monitoring and cooling system for detecting server hotspots, "
    "predicting temperature, controlling fan requirements and monitoring "
    "cooling effectiveness."
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard",
        "Model 1",
        "Model 2",
        "Model 3",
        "Model 4",
        "Model 5",
        "About"
    ]
)


if page == "Dashboard":

    st.header("System Dashboard")

    data = {}

    for model_number in range(1, 6):
        data[model_number] = load_data(model_number)

    missing_files = []

    for model_number, dataframe in data.items():
        if dataframe is None:
            missing_files.append(MODEL_FILES[model_number])

    if missing_files:

        st.error("The following files are missing:")

        for file in missing_files:
            st.write(f"- {file}")

        st.stop()

    model1 = data[1]
    model2 = data[2]
    model3 = data[3]
    model4 = data[4]
    model5 = data[5]

    temperature_columns = [
        "Server 1 Temperature C",
        "Server 2 Temperature C",
        "Server 3 Temperature C",
        "Server 4 Temperature C"
    ]

    highest_temperature = (
        model1[temperature_columns]
        .max()
        .max()
    )

    average_temperature = (
        model1[temperature_columns]
        .mean()
        .mean()
    )

    high_risk_count = int(
        (model3["Hotspot Risk"] >= 70).sum()
    )

    average_fan_speed = (
        model2["Required Fan Speed Percent"]
        .mean()
    )

    average_cooling = (
        model4["Cooling Effectiveness Percent"]
        .mean()
    )

    overheating_count = int(
        (model5["Overheating Warning"] == "OVERHEATING").sum()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Highest Temperature",
        f"{highest_temperature:.2f} °C"
    )

    col2.metric(
        "Average Temperature",
        f"{average_temperature:.2f} °C"
    )

    col3.metric(
        "High-Risk Records",
        high_risk_count
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Average Fan Requirement",
        f"{average_fan_speed:.2f}%"
    )

    col5.metric(
        "Cooling Effectiveness",
        f"{average_cooling:.2f}%"
    )

    col6.metric(
        "Overheating Records",
        overheating_count
    )

    st.subheader("Server Temperature Monitoring")

    st.line_chart(
        model1[temperature_columns].head(100)
    )

    st.subheader("Latest Server Data")

    st.dataframe(
        model1.tail(10),
        use_container_width=True
    )


elif page == "Model 1":

    st.header("Model 1 — Future Temperature Prediction")

    df = load_data(1)

    if df is None:
        st.error(
            f"Missing file: {MODEL_FILES[1]}"
        )
        st.stop()

    target = TARGETS[1]

    st.write(
        "This model predicts the future server temperature "
        "from current server and system conditions."
    )

    st.subheader("Dataset")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    model, features, mae, rmse, r2 = train_regression(
        df,
        target
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MAE",
        f"{mae:.3f}"
    )

    col2.metric(
        "RMSE",
        f"{rmse:.3f}"
    )

    col3.metric(
        "R² Score",
        f"{r2:.3f}"
    )

    st.subheader("Input Features")

    st.write(features)

    st.subheader("Feature Coefficients")

    coefficients = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_
    })

    coefficients["Absolute Importance"] = (
        coefficients["Coefficient"].abs()
    )

    coefficients = coefficients.sort_values(
        "Absolute Importance",
        ascending=False
    )

    st.dataframe(
        coefficients,
        use_container_width=True
    )


elif page == "Model 2":

    st.header("Model 2 — Required Fan Speed Prediction")

    df = load_data(2)

    if df is None:
        st.error(
            f"Missing file: {MODEL_FILES[2]}"
        )
        st.stop()

    target = TARGETS[2]

    st.write(
        "This model estimates the fan speed required "
        "to cool the server."
    )

    st.subheader("Dataset")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    model, features, mae, rmse, r2 = train_regression(
        df,
        target
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MAE",
        f"{mae:.3f}"
    )

    col2.metric(
        "RMSE",
        f"{rmse:.3f}"
    )

    col3.metric(
        "R² Score",
        f"{r2:.3f}"
    )

    st.subheader("Fan Speed Distribution")

    st.line_chart(
        df[target].head(100)
    )

    st.subheader("Feature Coefficients")

    coefficients = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_
    })

    coefficients["Absolute Importance"] = (
        coefficients["Coefficient"].abs()
    )

    coefficients = coefficients.sort_values(
        "Absolute Importance",
        ascending=False
    )

    st.dataframe(
        coefficients,
        use_container_width=True
    )


elif page == "Model 3":

    st.header("Model 3 — Hotspot Risk Detection")

    df = load_data(3)

    if df is None:
        st.error(
            f"Missing file: {MODEL_FILES[3]}"
        )
        st.stop()

    target = TARGETS[3]

    st.write(
        "This model calculates the probability/risk level "
        "of a server becoming a hotspot."
    )

    st.subheader("Dataset")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    average_risk = df[target].mean()
    maximum_risk = df[target].max()

    col1, col2 = st.columns(2)

    col1.metric(
        "Average Hotspot Risk",
        f"{average_risk:.2f}%"
    )

    col2.metric(
        "Maximum Hotspot Risk",
        f"{maximum_risk:.2f}%"
    )

    st.subheader("Hotspot Risk")

    st.line_chart(
        df[target].head(100)
    )

    if "Risk Level" in df.columns:

        st.subheader("Risk Level Distribution")

        risk_counts = df["Risk Level"].value_counts()

        st.bar_chart(
            risk_counts
        )

        st.dataframe(
            risk_counts.rename("Number of Records"),
            use_container_width=True
        )


elif page == "Model 4":

    st.header("Model 4 — Cooling Effectiveness")

    df = load_data(4)

    if df is None:
        st.error(
            f"Missing file: {MODEL_FILES[4]}"
        )
        st.stop()

    target = TARGETS[4]

    st.write(
        "This model measures how effectively the cooling "
        "system reduces server temperature."
    )

    st.subheader("Dataset")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    model, features, mae, rmse, r2 = train_regression(
        df,
        target
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Effectiveness",
        f"{df[target].mean():.2f}%"
    )

    col2.metric(
        "Maximum Effectiveness",
        f"{df[target].max():.2f}%"
    )

    col3.metric(
        "R² Score",
        f"{r2:.3f}"
    )

    st.subheader("Cooling Effectiveness")

    st.line_chart(
        df[target].head(100)
    )

    st.subheader("Feature Coefficients")

    coefficients = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_
    })

    coefficients["Absolute Importance"] = (
        coefficients["Coefficient"].abs()
    )

    coefficients = coefficients.sort_values(
        "Absolute Importance",
        ascending=False
    )

    st.dataframe(
        coefficients,
        use_container_width=True
    )


elif page == "Model 5":

    st.header("Model 5 — Overheating Warning")

    df = load_data(5)

    if df is None:
        st.error(
            f"Missing file: {MODEL_FILES[5]}"
        )
        st.stop()

    target = TARGETS[5]

    st.write(
        "This model provides an overheating warning "
        "based on server temperature and hotspot conditions."
    )

    st.subheader("Dataset")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    normal_count = int(
        (df[target] == "NORMAL").sum()
    )

    warning_count = int(
        (df[target] == "WARNING").sum()
    )

    overheating_count = int(
        (df[target] == "OVERHEATING").sum()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "NORMAL",
        normal_count
    )

    col2.metric(
        "WARNING",
        warning_count
    )

    col3.metric(
        "OVERHEATING",
        overheating_count
    )

    st.subheader("Warning Distribution")

    counts = df[target].value_counts()

    st.bar_chart(
        counts
    )

    st.dataframe(
        counts.rename("Number of Records"),
        use_container_width=True
    )


elif page == "About":

    st.header("About the Project")

    st.write(
        "The AI-Based Server Hotspot Cooling System is a prototype "
        "designed to monitor server temperatures, detect hotspots "
        "and dynamically support cooling decisions."
    )

    st.subheader("Five AI Models")

    st.markdown(
        """
        *Model 1 — Future Temperature Prediction*

        Predicts the future temperature of the server.

        *Model 2 — Required Fan Speed*

        Determines the required cooling fan speed.

        *Model 3 — Hotspot Risk*

        Determines the risk of a server becoming a hotspot.

        *Model 4 — Cooling Effectiveness*

        Measures the effectiveness of the cooling system.

        *Model 5 — Overheating Warning*

        Generates NORMAL, WARNING or OVERHEATING status.
        """
    )

    st.subheader("System Flow")

    st.code(
        """
ESP32 Temperature Sensors
            ↓
     Server Temperature
            ↓
       AI Models
            ↓
     Hotspot Detection
            ↓
    Required Fan Speed
            ↓
     Dynamic Cooling
            ↓
    Temperature Feedback
        """
    )

    st.info(
        "The included datasets are synthetic prototype data "
        "created for demonstration and model development. "
        "They are not measurements from a real data center."
    )
