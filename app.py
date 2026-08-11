import os
import uuid
import hashlib
import json
import base64
import smtplib
from io import BytesIO
from email.message import EmailMessage
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import folium
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Road Damage Detection System",
    page_icon="🚧",
    layout="wide"
)


# ============================================================
# FILES / FOLDERS
# ============================================================

MODEL_PATH = "best.pt"
SAVED_IMAGES_DIR = "saved_images"
REPORT_FILE = "damage_report.csv"
LOCATIONS_FILE = "damage_locations.json"

os.makedirs(SAVED_IMAGES_DIR, exist_ok=True)


# ============================================================
# LABELS
# ============================================================

LABELS = {
    "cracks": "Road Cracks Detected",
    "potholes": "Potholes Detected",
    "uneven surfaces": "Uneven Surface Detected",
    "unknown": "Unknown Damage"
}

RECOMMENDATIONS = {
    "cracks": "Medium Maintenance Required",
    "potholes": "Immediate Repair Required",
    "uneven surfaces": "Surface Inspection Needed",
    "unknown": "Further Inspection Needed"
}

# If the top two classes are too close, treat the prediction as uncertain
# instead of presenting a misleading confident classification.
UNCERTAINTY_GAP = 0.10


# ============================================================
# SESSION STATE
# ============================================================

if "report_ids" not in st.session_state:
    st.session_state.report_ids = {}

if "saved_reports" not in st.session_state:
    st.session_state.saved_reports = set()

if "save_messages" not in st.session_state:
    st.session_state.save_messages = set()

if "email_sent_reports" not in st.session_state:
    st.session_state.email_sent_reports = set()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


try:
    model = load_model()

except FileNotFoundError:
    st.error(
        "❌ The AI model file 'best.pt' was not found. "
        "Please place best.pt in the same folder as app.py."
    )
    st.stop()

except Exception as e:
    st.error(f"❌ Could not load the AI model: {e}")
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_locations():
    """Load saved GIS locations."""
    if not os.path.exists(LOCATIONS_FILE):
        return []

    try:
        with open(LOCATIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data if isinstance(data, list) else []

    except Exception:
        return []


def save_location(location_data):
    """Save one manually confirmed damage location."""
    locations = load_locations()

    locations.append(location_data)

    with open(
        LOCATIONS_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            locations,
            f,
            indent=4,
            ensure_ascii=False
        )


def get_color(prediction):
    if prediction == "cracks":
        return "red"

    if prediction == "potholes":
        return "orange"

    if prediction == "uneven surfaces":
        return "beige"

    return "green"


def get_report_status(report_id):
    """Return the current status of a report from the CSV file."""
    if not os.path.exists(REPORT_FILE):
        return "Pending"

    try:
        reports = pd.read_csv(REPORT_FILE)

        if "Status" not in reports.columns:
            return "Pending"

        matches = reports[
            reports["Report ID"].astype(str) == str(report_id)
        ]

        if matches.empty:
            return "Pending"

        status = matches.iloc[-1]["Status"]

        if pd.isna(status):
            return "Pending"

        return str(status)

    except Exception:
        return "Pending"


def image_to_data_uri(image_path):
    """Convert a saved image to a browser-embeddable data URI."""
    if not image_path or not os.path.exists(image_path):
        return None

    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

        return f"data:image/jpeg;base64,{encoded}"

    except Exception:
        return None


def image_to_thumbnail_data_uri(image):
    """Create a small embedded JPEG for persistent map popups."""
    try:
        thumbnail = image.copy()
        thumbnail.thumbnail((500, 350))

        buffer = BytesIO()

        thumbnail.save(
            buffer,
            format="JPEG",
            quality=75,
            optimize=True
        )

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return f"data:image/jpeg;base64,{encoded}"

    except Exception:
        return None


def get_popup_image(item):
    """Get the embedded image first, then fall back to local file."""
    image_data = item.get("Image Data")

    if image_data:
        return image_data

    return image_to_data_uri(
        item.get("Image Path")
    )


# ============================================================
# EMAIL NOTIFICATION
# ============================================================

def send_save_notification(
    report_data,
    image_path
):
    """
    Send an email notification after a report is successfully saved.
    Credentials are read from Streamlit Secrets and are never stored
    directly in the application code.
    """

    try:

        gmail_address = st.secrets["GMAIL_ADDRESS"]
        gmail_app_password = st.secrets["GMAIL_APP_PASSWORD"]
        notification_email = st.secrets["NOTIFICATION_EMAIL"]

    except Exception:

        return False, (
            "Gmail notification is not configured. "
            "Add GMAIL_ADDRESS, GMAIL_APP_PASSWORD, "
            "and NOTIFICATION_EMAIL to Streamlit Secrets."
        )

    try:

        message = EmailMessage()

        message["Subject"] = (
            f"🚨 New Road Damage Report - "
            f"{report_data.get('Report ID', 'Unknown')}"
        )

        message["From"] = gmail_address
        message["To"] = notification_email

        message.set_content(
            f"""New Road Damage Report

Report ID: {report_data.get('Report ID', 'Unknown')}
Image: {report_data.get('Image', 'Unknown')}
Prediction: {report_data.get('Prediction', 'Unknown')}
Confidence: {float(report_data.get('Confidence', 0)):.2%}
Risk: {report_data.get('Risk', 'Unknown')}
Recommendation: {report_data.get('Recommendation', 'Unknown')}

Latitude: {report_data.get('Latitude', 'N/A')}
Longitude: {report_data.get('Longitude', 'N/A')}

Time: {report_data.get('Time', 'Unknown')}

The report was manually saved by a system user.
The damage image is attached to this email.
"""
        )

        if image_path and os.path.exists(image_path):

            with open(
                image_path,
                "rb"
            ) as image_file:

                image_bytes = image_file.read()

            message.add_attachment(
                image_bytes,
                maintype="image",
                subtype="jpeg",
                filename=os.path.basename(
                    image_path
                )
            )

        with smtplib.SMTP(
            "smtp.gmail.com",
            587,
            timeout=30
        ) as server:

            server.starttls()

            server.login(
                gmail_address,
                gmail_app_password
            )

            server.send_message(
                message
            )

        return True, "Email notification sent successfully."

    except Exception as e:

        return False, (
            f"Report was saved, but the email notification "
            f"could not be sent: {e}"
        )


# ============================================================
# TITLE
# ============================================================

st.title("🚧 Road Damage Detection System")

st.subheader(
    "AI-Based Road Damage Classification with GIS Integration"
)

st.write(
    "Upload one or more road images. "
    "The AI analyzes the images and you can manually save "
    "the selected reports together with their GPS location."
)


# ============================================================
# LOGO
# ============================================================

if os.path.exists("logo.png"):
    st.image("logo.png", width=150)


# ============================================================
# LOCATION
# ============================================================

st.markdown("---")
st.subheader("📍 Current GPS Location")

st.write(
    "Allow location access in your browser so the saved damage "
    "report can be associated with its real location."
)

location = streamlit_geolocation()

lat_user = location.get("latitude")
lon_user = location.get("longitude")

if lat_user is not None and lon_user is not None:

    st.success(
        f"📍 Location detected: "
        f"{lat_user:.6f}, {lon_user:.6f}"
    )

else:

    st.warning(
        "⚠️ GPS location is not available yet. "
        "Allow location access and try again."
    )


# ============================================================
# UPLOAD IMAGES
# ============================================================

uploaded_files = st.file_uploader(
    "📂 Upload Road Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)


# ============================================================
# CAMERA
# ============================================================

camera_image = st.camera_input(
    "📷 Or take a road photo"
)


# ============================================================
# PREPARE IMAGES
# ============================================================

images = []

if uploaded_files:

    for file in uploaded_files:

        try:

            image_bytes = file.getvalue()

            image = Image.open(file).convert("RGB")

            image_key = hashlib.md5(
                image_bytes
            ).hexdigest()

            images.append(
                (
                    file.name,
                    image,
                    image_key
                )
            )

        except Exception:

            st.error(
                f"❌ Could not open {file.name}"
            )

elif camera_image is not None:

    try:

        image_bytes = camera_image.getvalue()

        image = Image.open(
            camera_image
        ).convert("RGB")

        image_key = hashlib.md5(
            image_bytes
        ).hexdigest()

        images.append(
            (
                "Camera Image",
                image,
                image_key
            )
        )

    except Exception:

        st.error(
            "❌ Could not open the camera image."
        )


# ============================================================
# ANALYSIS
# ============================================================

if images:

    st.markdown("---")

    st.subheader(
        f"🔍 Analysis Results ({len(images)} image(s))"
    )

    all_results = []

    progress = st.progress(0)

    for index, (image_name, image, image_key) in enumerate(images):

        st.markdown("---")

        st.subheader(
            f"📷 {image_name}"
        )

        st.image(
            image,
            caption=image_name,
            use_container_width=True
        )

        # ----------------------------------------------------
        # STABLE REPORT ID
        # ----------------------------------------------------

        if image_key not in st.session_state.report_ids:

            st.session_state.report_ids[image_key] = (
                str(uuid.uuid4())[:8]
            )

        report_id = st.session_state.report_ids[
            image_key
        ]

        # ----------------------------------------------------
        # AI PREDICTION
        # ----------------------------------------------------

        try:

            with st.spinner(
                f"Analyzing {image_name}..."
            ):

                results = model(image)

            result = results[0]

            if result.probs is None:

                st.error(
                    f"❌ The model did not return "
                    f"classification probabilities for {image_name}."
                )

                continue

            top1 = int(
                result.probs.top1
            )

            confidence = float(
                result.probs.top1conf
            )

            prediction = result.names[top1]

            # Get confidence for every class so we can diagnose
            # confusion between Cracks and Uneven Surface.
            class_probabilities = []

            if result.probs is not None:

                for class_index, probability in enumerate(
                    result.probs.data.tolist()
                ):

                    class_name = result.names.get(
                        class_index,
                        str(class_index)
                    )

                    class_probabilities.append(
                        {
                            "Class": LABELS.get(
                                class_name,
                                class_name
                            ),
                            "Confidence": float(
                                probability
                            )
                        }
                    )

                class_probabilities = sorted(
                    class_probabilities,
                    key=lambda item: item["Confidence"],
                    reverse=True
                )

        except Exception as e:

            st.error(
                f"❌ Error analyzing {image_name}: {e}"
            )

            continue

        prediction_text = LABELS.get(
            prediction,
            prediction
        )

        recommendation = RECOMMENDATIONS.get(
            prediction,
            "Further Inspection Needed"
        )

        color = get_color(
            prediction
        )

        # ----------------------------------------------------
        # UNCERTAINTY CHECK
        # ----------------------------------------------------

        is_uncertain = False
        second_class = None
        second_confidence = 0.0
        confidence_gap = confidence

        if len(class_probabilities) >= 2:

            second_class = class_probabilities[1]["Class"]

            second_confidence = class_probabilities[1]["Confidence"]

            confidence_gap = (
                confidence - second_confidence
            )

            is_uncertain = (
                confidence_gap < UNCERTAINTY_GAP
            )

        if is_uncertain:

            prediction_text = "⚠️ Uncertain Classification"

            recommendation = (
                f"AI is uncertain between "
                f"{LABELS.get(prediction, prediction)} "
                f"and {second_class}. "
                f"Manual inspection is recommended."
            )

            color = "orange"

        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------

        if is_uncertain:

            risk = "Needs Review"

        elif confidence > 0.90:

            risk = "High Risk"

        elif confidence > 0.70:

            risk = "Medium Risk"

        else:

            risk = "Low Risk"

        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        current_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        st.markdown(
            "### 📊 Detection Result"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Prediction",
                prediction_text
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence:.2%}"
            )

        st.progress(
            confidence
        )

        if is_uncertain:

            st.warning(
                f"⚠️ The AI is uncertain about this image. "
                f"The top two classes are only "
                f"{confidence_gap:.2%} apart: "
                f"{prediction_text}."
            )

            st.info(
                "The system will not treat this as a confirmed "
                "damage classification. Manual review is recommended."
            )

        # ----------------------------------------------------
        # ALL CLASS PROBABILITIES
        # ----------------------------------------------------

        with st.expander(
            "🔬 AI Classification Details"
        ):

            probability_df = pd.DataFrame(
                class_probabilities
            )

            probability_df["Confidence"] = (
                probability_df["Confidence"]
                .map(
                    lambda value: f"{value:.2%}"
                )
            )

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "This section shows how confident the AI is "
                "about each of the four classes."
            )

            if len(class_probabilities) >= 2:

                st.write(
                    f"**Top-two confidence gap:** "
                    f"{confidence_gap:.2%}"
                )

                st.write(
                    f"**Uncertainty threshold:** "
                    f"{UNCERTAINTY_GAP:.0%}"
                )

        st.markdown(
            f"**Risk Level:** {risk}"
        )

        st.markdown(
            f"**Recommendation:** {recommendation}"
        )

        st.markdown(
            f"**Detection Time:** {current_time}"
        )

        st.markdown(
            f"**Report ID:** `{report_id}`"
        )

        # ----------------------------------------------------
        # REPORT DATA
        # ----------------------------------------------------

        report_data = {

            "Report ID": report_id,

            "Image": image_name,

            "Prediction": prediction_text,

            "Confidence": round(
                confidence,
                4
            ),

            "Risk": risk,

            "Recommendation": recommendation,

            "Latitude": lat_user,

            "Longitude": lon_user,

            "Image Path": os.path.join(
                SAVED_IMAGES_DIR,
                f"{report_id}.jpg"
            ),

            "Time": current_time
        }

        all_results.append(
            report_data
        )

        # ----------------------------------------------------
        # SAVED MESSAGE
        # ----------------------------------------------------

        if report_id in st.session_state.save_messages:

            st.success(
                f"✅ Report {report_id} saved successfully."
            )

        # ----------------------------------------------------
        # SAVE BUTTON
        # ----------------------------------------------------

        save_button = st.button(
            "💾 Save Report + GPS Location",
            key=f"save_button_{image_key}",
            use_container_width=True
        )

        if save_button:

            if report_id in st.session_state.saved_reports:

                st.info(
                    "ℹ️ This report has already been saved."
                )

            elif lat_user is None or lon_user is None:

                st.error(
                    "❌ Cannot save this report because "
                    "the GPS location has not been detected. "
                    "Allow location access and try again."
                )

            else:

                # --------------------------------------------
                # SAVE IMAGE
                # --------------------------------------------

                image_path = os.path.join(
                    SAVED_IMAGES_DIR,
                    f"{report_id}.jpg"
                )

                image.save(
                    image_path,
                    format="JPEG",
                    quality=95
                )

                # --------------------------------------------
                # SAVE CSV REPORT
                # --------------------------------------------

                report_df = pd.DataFrame(
                    [report_data]
                )

                if os.path.exists(
                    REPORT_FILE
                ):

                    old_df = pd.read_csv(
                        REPORT_FILE
                    )

                    combined_df = pd.concat(
                        [
                            old_df,
                            report_df
                        ],
                        ignore_index=True
                    )

                else:

                    combined_df = report_df

                combined_df.to_csv(
                    REPORT_FILE,
                    index=False
                )

                # --------------------------------------------
                # SAVE GIS LOCATION
                # --------------------------------------------

                # Create a small embedded copy of the image.
                # This makes the image available inside the map popup
                # even after the app is restarted or deployed.
                image_data = image_to_thumbnail_data_uri(
                    image
                )

                location_data = {

                    "Report ID": report_id,

                    "Image": image_name,

                    "Prediction": prediction_text,

                    "Risk": risk,

                    "Confidence": round(
                        confidence,
                        4
                    ),

                    "Latitude": lat_user,

                    "Longitude": lon_user,

                    "Color": color,

                    "Image Path": image_path,

                    "Image Data": image_data,

                    "Status": "Pending",

                    "Time": current_time
                }

                save_location(
                    location_data
                )

                # --------------------------------------------
                # UPDATE SESSION
                # --------------------------------------------

                st.session_state.saved_reports.add(
                    report_id
                )

                st.session_state.save_messages.add(
                    report_id
                )

                # ------------------------------------------------
                # SEND EMAIL NOTIFICATION
                # ------------------------------------------------

                if report_id not in st.session_state.email_sent_reports:

                    email_sent, email_message = send_save_notification(
                        report_data,
                        image_path
                    )

                    if email_sent:

                        st.session_state.email_sent_reports.add(
                            report_id
                        )

                        st.success(
                            "📧 Email notification sent successfully."
                        )

                    else:

                        st.warning(
                            f"⚠️ {email_message}"
                        )

                st.success(
                    f"✅ Report {report_id} and GPS location saved successfully."
                )

        progress.progress(
            (index + 1) / len(images)
        )


    # ========================================================
    # CURRENT ANALYSIS SUMMARY
    # ========================================================

    if all_results:

        st.markdown("---")

        st.subheader(
            "📋 Current Analysis Summary"
        )

        st.dataframe(
            pd.DataFrame(all_results),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# GIS MAP
# ============================================================

st.markdown("---")

st.subheader(
    "🗺️ Saved Road Damage Map"
)

saved_locations = load_locations()

if saved_locations:

    # --------------------------------------------------------
    # MAP CENTER
    # --------------------------------------------------------

    valid_locations = [
        item for item in saved_locations
        if item.get("Latitude") is not None
        and item.get("Longitude") is not None
        and get_report_status(item.get("Report ID")) != "Completed"
    ]

    if valid_locations:

        latest = valid_locations[-1]

        map_center = [
            float(latest["Latitude"]),
            float(latest["Longitude"])
        ]

        m = folium.Map(
            location=map_center,
            zoom_start=13
        )

        # ----------------------------------------------------
        # ADD SAVED DAMAGE POINTS
        # ----------------------------------------------------

        for item in valid_locations:

            lat = float(
                item["Latitude"]
            )

            lon = float(
                item["Longitude"]
            )

            color = item.get(
                "Color",
                "red"
            )

            image_uri = get_popup_image(
                item
            )

            image_html = ""

            if image_uri:
                image_html = (
                    f'<br><img src="{image_uri}" '
                    f'alt="Saved road damage image" '
                    f'style="width:280px; max-height:200px; '
                    f'object-fit:contain; border-radius:8px; '
                    f'margin-top:8px;">'
                )

            popup_text = (
                f"<div style='width:300px; font-family:Arial;'>"
                f"<h4 style='margin-bottom:8px;'>"
                f"Road Damage Report"
                f"</h4>"
                f"<b>Report ID:</b> {item.get('Report ID', 'Unknown')}<br>"
                f"<b>Damage:</b> {item.get('Prediction', 'Unknown')}<br>"
                f"<b>Risk:</b> {item.get('Risk', 'Unknown')}<br>"
                f"<b>Confidence:</b> {item.get('Confidence', 'Unknown')}<br>"
                f"<b>Time:</b> {item.get('Time', 'Unknown')}"
                f"{image_html}"
                f"</div>"
            )

            folium.CircleMarker(
                location=[
                    lat,
                    lon
                ],
                radius=9,
                popup=folium.Popup(
                    popup_text,
                    max_width=300
                ),
                tooltip=item.get(
                    "Prediction",
                    "Road Damage"
                ),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7
            ).add_to(m)

        # ----------------------------------------------------
        # DISPLAY MAP
        # ----------------------------------------------------

        st_folium(
            m,
            width=1100,
            height=550,
            returned_objects=[]
        )

        st.caption(
            "💡 Click any damage marker on the map to view "
            "the report data and the saved damage image."
        )

    else:

        st.info(
            "No valid GPS locations are available."
        )

else:

    st.info(
        "No active damage locations are currently shown. "
        "Completed reports are hidden from the monitoring map."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Road Damage Detection System | "
    "Developed by Rudaina, Maryam, AlBalja"
)

