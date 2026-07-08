import streamlit as st
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
import pandas as pd
import folium
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation
import json
import os

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="Road Damage Detection System",
    layout="centered"
)

# ================= LOAD MODEL =================

model = YOLO("best.pt")

# ================= SESSION STATE =================

if "last_saved" not in st.session_state:

    st.session_state.last_saved = None

# ================= TITLE =================

st.title("🚧 Road Damage Detection System")

st.subheader(
    "AI-Based Road Damage Classification with GIS Integration"
)

st.write(
    "Upload a road image or use the camera for AI road damage detection."
)

# ================= LOGO =================

st.image("logo.png", width=150)

# ================= FILE UPLOAD =================

uploaded_file = st.file_uploader(
    "📂 Upload Road Image",
    type=["jpg", "png", "jpeg"]
)

# ================= CAMERA =================

camera_image = st.camera_input(
    "📷 Take a Road Photo"
)

# ================= LABELS =================

labels = {
    "cracks": "Road Cracks Detected",
    "potholes": "Potholes Detected",
    "uneven surfaces": "Uneven Surface Detected",
    "unknown": "Unknown Damage"
}

# ================= MAIN =================

if uploaded_file is not None or camera_image is not None:

    # ================= IMAGE SOURCE =================

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

    else:

        image = Image.open(camera_image)

    # ================= SHOW IMAGE =================

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # ================= AI PREDICTION =================

    results = model(image)

    top1 = results[0].probs.top1

    confidence = float(results[0].probs.top1conf)

    prediction = results[0].names[top1]

    prediction_text = labels.get(
        prediction,
        prediction
    )

    # ================= COLORS =================

    if prediction == "cracks":

        color = "red"

    elif prediction == "potholes":

        color = "orange"

    elif prediction == "uneven surfaces":

        color = "beige"

    else:

        color = "green"

    # ================= RISK LEVEL =================

    if confidence > 0.90:

        risk = "High Risk"

    elif confidence > 0.70:

        risk = "Medium Risk"

    else:

        risk = "Low Risk"

    # ================= TIME =================

    current_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M"
    )

    # ================= RESULTS =================

    st.markdown("---")

    st.markdown(
        f"## Prediction: :{color}[{prediction_text}]"
    )

    st.markdown(
        f"### Confidence Score: {confidence:.2f}"
    )

    st.progress(confidence)

    st.markdown(
        f"### Risk Level: :{color}[{risk}]"
    )

    st.markdown(
        f"### Detection Time: {current_time}"
    )

    # ================= REPORT TABLE =================

    result_data = {
        "Prediction": [prediction_text],
        "Confidence": [confidence],
        "Risk": [risk],
        "Time": [current_time]
    }

    df = pd.DataFrame(result_data)

    st.markdown("---")

    st.subheader("📊 Detection Report")

    st.dataframe(df)

    # ================= GIS SECTION =================

    st.markdown("---")

    st.subheader("📍 GIS Damage Monitoring Map")

    st.write(
        "Road damage locations are automatically saved and highlighted."
    )

    # ================= GET USER LOCATION =================

    location = streamlit_geolocation()

    # ================= LOCATION FOUND =================

    if location["latitude"] is not None:

        lat_user = location["latitude"]

        lon_user = location["longitude"]

        st.success(
            f"""
            Current Location Detected

            Latitude: {lat_user}

            Longitude: {lon_user}
            """
        )

        # ================= LOAD OLD LOCATIONS =================

        if os.path.exists("damage_locations.json"):

            with open("damage_locations.json", "r") as f:

                saved_locations = json.load(f)

        else:

            saved_locations = []

        # ================= CREATE MAP =================

        m = folium.Map(
            location=[lat_user, lon_user],
            zoom_start=15
        )

        # ================= SHOW SAVED DAMAGES =================

        for item in saved_locations:

            folium.Circle(
                location=[
                    item.get("lat", lat_user),
                    item.get("lon", lon_user)
                ],
                radius=30,
                popup=f"{item.get('prediction', 'Unknown')}\nRisk: {item.get('risk', 'Unknown')}",
                color=item.get("color", "red"),
                fill=True,
                fill_color=item.get("color", "red"),
                fill_opacity=0.4
            ).add_to(m)

            folium.Marker(
                [
                    item.get("lat", lat_user),
                    item.get("lon", lon_user)
                ],
                tooltip=item.get(
                    "prediction",
                    "Unknown"
                )
            ).add_to(m)

        # ================= CURRENT DAMAGE =================

        folium.Circle(
            location=[lat_user, lon_user],
            radius=30,
            popup=prediction_text,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.5
        ).add_to(m)

        folium.Marker(
            [lat_user, lon_user],
            popup=f"{prediction_text}\nRisk: {risk}",
            tooltip="Detected Damage"
        ).add_to(m)

        # ================= NEW DAMAGE =================

        new_damage = {
            "lat": lat_user,
            "lon": lon_user,
            "prediction": prediction_text,
            "risk": risk,
            "color": color,
            "time": current_time
        }

        # ================= PREVENT DUPLICATES =================

        current_damage = (
            lat_user,
            lon_user,
            prediction_text,
            risk
        )

        if st.session_state.last_saved != current_damage:

            saved_locations.append(new_damage)

            with open("damage_locations.json", "w") as f:

                json.dump(saved_locations, f)

            st.session_state.last_saved = current_damage

        # ================= SHOW MAP =================

        st_folium(
            m,
            width=700,
            height=500,
            returned_objects=[]
        )

        # ================= SAVE MAP =================

        m.save("saved_damage_map.html")

        # ================= DOWNLOAD MAP =================

        with open("saved_damage_map.html", "rb") as map_file:

            st.download_button(
                label="🗺 Download Damage Map",
                data=map_file,
                file_name="saved_damage_map.html",
                mime="text/html"
            )

        # ================= REPORT DATA =================

        report_data = {
            "Prediction": prediction_text,
            "Confidence": round(confidence, 2),
            "Risk": risk,
            "Latitude": lat_user,
            "Longitude": lon_user,
            "Time": current_time
        }

        report_df = pd.DataFrame([report_data])

        # ================= SAVE REPORT =================

        if os.path.exists("damage_report.csv"):

            old_df = pd.read_csv(
                "damage_report.csv"
            )

            new_df = pd.concat(
                [old_df, report_df],
                ignore_index=True
            )

            new_df.to_csv(
                "damage_report.csv",
                index=False
            )

        else:

            report_df.to_csv(
                "damage_report.csv",
                index=False
            )

        # ================= DOWNLOAD REPORT =================

        with open("damage_report.csv", "rb") as file:

            st.download_button(
                label="📥 Download Full Report",
                data=file,
                file_name="damage_report.csv",
                mime="text/csv"
            )

    # ================= LOCATION ERROR =================

    else:

        st.warning(
            "Please allow location access from your browser."
        )

# ================= FOOTER =================

st.markdown("---")

st.caption("Developed by Rudaina, Maryam, AlBalja")