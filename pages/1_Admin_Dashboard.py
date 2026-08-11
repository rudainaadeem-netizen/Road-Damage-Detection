import os
import json
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import folium
import matplotlib.pyplot as plt
from streamlit_folium import st_folium


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Admin Dashboard | Road Damage Detection",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

REPORT_FILE = "damage_report.csv"
LOCATIONS_FILE = "damage_locations.json"
SAVED_IMAGES_DIR = "saved_images"


# ============================================================
# ADMIN AUTHENTICATION
# ============================================================

st.title("📊 Admin Dashboard")

st.write(
    "Administrative monitoring and management of saved road damage reports."
)

st.markdown("---")

# For local development:
# PowerShell:
#   $env:ADMIN_PASSWORD="your_password"
# For Streamlit Cloud, add ADMIN_PASSWORD to Secrets.
configured_password = os.environ.get("ADMIN_PASSWORD", "")

try:
    if not configured_password:
        configured_password = st.secrets.get("ADMIN_PASSWORD", "")
except Exception:
    pass

if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

if not st.session_state.admin_authenticated:

    if not configured_password:

        st.warning(
            "🔐 Admin password is not configured yet."
        )

        st.info(
            "For local testing, set an ADMIN_PASSWORD environment "
            "variable before running Streamlit."
        )

        st.code(
            '$env:ADMIN_PASSWORD="YourPasswordHere"',
            language="powershell"
        )

        st.stop()

    password = st.text_input(
        "🔑 Admin Password",
        type="password"
    )

    if st.button(
        "Login",
        type="primary"
    ):

        if password == configured_password:

            st.session_state.admin_authenticated = True
            st.rerun()

        else:

            st.error(
                "❌ Incorrect password."
            )

    st.stop()


# ============================================================
# LOGOUT
# ============================================================

if st.sidebar.button("🔒 Logout"):

    st.session_state.admin_authenticated = False
    st.rerun()


st.sidebar.success(
    "Admin authenticated"
)


# ============================================================
# LOAD REPORTS
# ============================================================

if not os.path.exists(REPORT_FILE):

    st.info(
        "No saved reports are available yet."
    )

    st.stop()


try:

    df = pd.read_csv(
        REPORT_FILE
    )

except Exception as e:

    st.error(
        f"Could not read the report database: {e}"
    )

    st.stop()


# ============================================================
# PREPARE DATA
# ============================================================

# Older reports may not have Status.
if "Status" not in df.columns:

    df["Status"] = "Pending"


df["Status"] = df["Status"].fillna(
    "Pending"
)

if "Priority" not in df.columns:
    df["Priority"] = "Medium"

df["Priority"] = df["Priority"].fillna("Medium")

if "Admin Notes" not in df.columns:
    df["Admin Notes"] = ""

df["Admin Notes"] = df["Admin Notes"].fillna("")

if "Risk" in df.columns:

    df["Risk"] = df["Risk"].fillna(
        "Unknown"
    )

if "Prediction" in df.columns:

    df["Prediction"] = df["Prediction"].fillna(
        "Unknown"
    )


# ============================================================
# SMART PRIORITY SUGGESTION
# ============================================================

def suggested_priority(row):
    """Suggest priority from AI risk/uncertainty."""
    risk = str(row.get("Risk", "")).lower()
    prediction = str(row.get("Prediction", "")).lower()

    if "needs review" in risk or "uncertain" in prediction:
        return "High"

    if "high" in risk:
        return "High"

    if "medium" in risk:
        return "Medium"

    if "low" in risk:
        return "Low"

    return "Medium"


# Existing reports that have no meaningful priority get a
# suggested priority based on their AI result.
for row_index in df.index:
    current_priority = str(df.loc[row_index, "Priority"]).strip()

    if (
        current_priority == ""
        or current_priority.lower() == "nan"
    ):
        df.loc[row_index, "Priority"] = suggested_priority(
            df.loc[row_index]
        )


# ============================================================
# DASHBOARD METRICS
# ============================================================


total_reports = len(df)

cracks_count = int(
    df["Prediction"]
    .astype(str)
    .str.contains(
        "Crack",
        case=False,
        na=False
    )
    .sum()
)

potholes_count = int(
    df["Prediction"]
    .astype(str)
    .str.contains(
        "Pothole",
        case=False,
        na=False
    )
    .sum()
)

high_risk_count = int(
    df["Risk"]
    .astype(str)
    .str.contains(
        "High",
        case=False,
        na=False
    )
    .sum()
)

needs_review_count = int(
    df["Risk"]
    .astype(str)
    .str.contains(
        "Needs Review",
        case=False,
        na=False
    )
    .sum()
)

pending_count = int(
    df["Status"]
    .astype(str)
    .eq("Pending")
    .sum()
)


st.subheader("📈 Overview")

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.metric(
        "Total Reports",
        total_reports
    )

with c2:
    st.metric(
        "Cracks",
        cracks_count
    )

with c3:
    st.metric(
        "Potholes",
        potholes_count
    )

with c4:
    st.metric(
        "High Risk",
        high_risk_count
    )

with c5:
    st.metric(
        "Pending",
        pending_count
    )

with c6:
    st.metric(
        "Needs Review",
        needs_review_count
    )


# ============================================================
# DASHBOARD CHARTS
# ============================================================

st.markdown("---")

st.subheader("📊 Dashboard Analytics")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:

    st.markdown("#### Damage Types")

    damage_counts = (
        df["Prediction"]
        .astype(str)
        .value_counts()
    )

    if len(damage_counts) > 0:

        fig1, ax1 = plt.subplots(
            figsize=(7, 4)
        )

        damage_counts.plot(
            kind="bar",
            ax=ax1
        )

        ax1.set_xlabel("Damage Type")
        ax1.set_ylabel("Number of Reports")
        ax1.set_title("Road Damage Distribution")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()

        st.pyplot(
            fig1,
            use_container_width=True
        )

        plt.close(fig1)

    else:

        st.info("No damage data available.")


with chart_col2:

    st.markdown("#### Report Status")

    status_counts = (
        df["Status"]
        .astype(str)
        .value_counts()
    )

    if len(status_counts) > 0:

        fig2, ax2 = plt.subplots(
            figsize=(7, 4)
        )

        status_counts.plot(
            kind="pie",
            autopct="%1.0f%%",
            ax=ax2
        )

        ax2.set_ylabel("")
        ax2.set_title("Report Status Distribution")
        plt.tight_layout()

        st.pyplot(
            fig2,
            use_container_width=True
        )

        plt.close(fig2)

    else:

        st.info("No status data available.")


chart_col3, chart_col4 = st.columns(2)

with chart_col3:

    st.markdown("#### Priority Distribution")

    priority_counts = (
        df["Priority"]
        .astype(str)
        .value_counts()
        .reindex(
            ["High", "Medium", "Low"],
            fill_value=0
        )
    )

    fig3, ax3 = plt.subplots(
        figsize=(7, 4)
    )

    priority_counts.plot(
        kind="bar",
        ax=ax3
    )

    ax3.set_xlabel("Priority")
    ax3.set_ylabel("Number of Reports")
    ax3.set_title("Report Priority Distribution")
    plt.tight_layout()

    st.pyplot(
        fig3,
        use_container_width=True
    )

    plt.close(fig3)


with chart_col4:

    st.markdown("#### Risk Distribution")

    risk_counts = (
        df["Risk"]
        .astype(str)
        .value_counts()
        .reindex(
            ["High Risk", "Medium Risk", "Low Risk"],
            fill_value=0
        )
    )

    fig4, ax4 = plt.subplots(
        figsize=(7, 4)
    )

    risk_counts.plot(
        kind="bar",
        ax=ax4
    )

    ax4.set_xlabel("Risk Level")
    ax4.set_ylabel("Number of Reports")
    ax4.set_title("Risk Level Distribution")
    plt.tight_layout()

    st.pyplot(
        fig4,
        use_container_width=True
    )

    plt.close(fig4)


# ============================================================
# FILTERS
# ============================================================

st.markdown("---")

st.subheader("🔎 Report Filters")

f1, f2, f3 = st.columns(3)

with f1:

    prediction_options = [
        "All",
        "Road Cracks Detected",
        "Potholes Detected",
        "Uneven Surface Detected",
        "Unknown Damage"
    ]

    selected_prediction = st.selectbox(
        "Damage Type",
        prediction_options
    )

with f2:

    risk_options = [
        "All"
    ] + sorted(
        df["Risk"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_risk = st.selectbox(
        "Risk Level",
        risk_options
    )

with f3:

    status_options = [
        "All",
        "Pending",
        "In Progress",
        "Completed"
    ]

    selected_status = st.selectbox(
        "Status",
        status_options
    )

selected_priority = st.selectbox(
    "Priority",
    [
        "All",
        "High",
        "Medium",
        "Low"
    ]
)


# ============================================================
# DATE & LOCATION FILTERS
# ============================================================

st.markdown("---")

st.subheader("📅🌍 Date & Location Filters")

date_col1, date_col2 = st.columns(2)

today = datetime.now().date()

with date_col1:

    date_filter_enabled = st.checkbox(
        "Enable date filter",
        value=False
    )

with date_col2:

    location_filter_enabled = st.checkbox(
        "Enable location filter",
        value=False
    )

if date_filter_enabled:

    default_start = today - timedelta(days=7)

    selected_start_date = st.date_input(
        "Start Date",
        value=default_start
    )

    selected_end_date = st.date_input(
        "End Date",
        value=today
    )

    if selected_start_date > selected_end_date:

        st.error(
            "❌ Start Date cannot be later than End Date."
        )


if location_filter_enabled:

    location_col1, location_col2 = st.columns(2)

    with location_col1:

        min_latitude = st.number_input(
            "Minimum Latitude",
            value=0.0,
            format="%.6f"
        )

        min_longitude = st.number_input(
            "Minimum Longitude",
            value=0.0,
            format="%.6f"
        )

    with location_col2:

        max_latitude = st.number_input(
            "Maximum Latitude",
            value=90.0,
            format="%.6f"
        )

        max_longitude = st.number_input(
            "Maximum Longitude",
            value=180.0,
            format="%.6f"
        )

    if min_latitude > max_latitude:

        st.error(
            "❌ Minimum Latitude cannot be greater than Maximum Latitude."
        )

    if min_longitude > max_longitude:

        st.error(
            "❌ Minimum Longitude cannot be greater than Maximum Longitude."
        )


# ============================================================
# SEARCH BY REPORT ID
# ============================================================

st.markdown("---")

search_report_id = st.text_input(
    "🔎 Search by Report ID",
    placeholder="Enter Report ID, e.g. A82F91C3"
)

filtered_df = df.copy()

# ------------------------------------------------------------
# DATE FILTER
# ------------------------------------------------------------

if date_filter_enabled:

    if selected_start_date <= selected_end_date:

        if "Time" in filtered_df.columns:

            report_dates = pd.to_datetime(
                filtered_df["Time"],
                errors="coerce"
            ).dt.date

            filtered_df = filtered_df[
                report_dates.between(
                    selected_start_date,
                    selected_end_date
                )
            ]


# ------------------------------------------------------------
# LOCATION FILTER
# ------------------------------------------------------------

if location_filter_enabled:

    if (
        min_latitude <= max_latitude
        and min_longitude <= max_longitude
    ):

        if (
            "Latitude" in filtered_df.columns
            and "Longitude" in filtered_df.columns
        ):

            filtered_df["Latitude"] = pd.to_numeric(
                filtered_df["Latitude"],
                errors="coerce"
            )

            filtered_df["Longitude"] = pd.to_numeric(
                filtered_df["Longitude"],
                errors="coerce"
            )

            filtered_df = filtered_df[
                filtered_df["Latitude"].between(
                    min_latitude,
                    max_latitude
                )
                &
                filtered_df["Longitude"].between(
                    min_longitude,
                    max_longitude
                )
            ]


if search_report_id.strip():

    search_value = search_report_id.strip().lower()

    filtered_df = filtered_df[
        filtered_df["Report ID"]
        .astype(str)
        .str.lower()
        .str.contains(
            search_value,
            na=False
        )
    ]

if selected_prediction != "All":

    filtered_df = filtered_df[
        filtered_df["Prediction"]
        == selected_prediction
    ]

if selected_risk != "All":

    filtered_df = filtered_df[
        filtered_df["Risk"]
        == selected_risk
    ]

if selected_status != "All":

    filtered_df = filtered_df[
        filtered_df["Status"]
        == selected_status
    ]

if selected_priority != "All":

    filtered_df = filtered_df[
        filtered_df["Priority"]
        == selected_priority
    ]


st.write(
    f"Showing **{len(filtered_df)}** report(s) after applying the current filters."
)

active_filters = []

if date_filter_enabled:
    active_filters.append(
        f"Date: {selected_start_date} → {selected_end_date}"
    )

if location_filter_enabled:
    active_filters.append(
        f"Location: "
        f"Lat {min_latitude:.4f}–{max_latitude:.4f}, "
        f"Lon {min_longitude:.4f}–{max_longitude:.4f}"
    )

if search_report_id.strip():
    active_filters.append(
        f"Report ID: {search_report_id.strip()}"
    )

if active_filters:

    st.info(
        "🔎 Active filters: " + " | ".join(active_filters)
    )

if search_report_id.strip():

    if len(filtered_df) == 0:

        st.warning(
            f"No report was found for Report ID: `{search_report_id.strip()}`"
        )

    elif len(filtered_df) > 1:

        st.info(
            "More than one matching Report ID was found. "
            "Select the required report below."
        )


# ============================================================
# REPORT MANAGEMENT
# ============================================================

st.markdown("---")

st.subheader("📝 Manage Reports")

if len(filtered_df) > 0:

    report_ids = filtered_df[
        "Report ID"
    ].astype(str).tolist()

    if search_report_id.strip() and len(report_ids) == 1:

        selected_report_id = report_ids[0]

        st.success(
            f"Report `{selected_report_id}` found."
        )

    else:

        selected_report_id = st.selectbox(
            "Select a report to manage",
            report_ids
        )

    selected_index = df.index[
        df["Report ID"].astype(str)
        == selected_report_id
    ].tolist()[0]

    selected_row = df.loc[
        selected_index
    ]

    m1, m2 = st.columns([2, 1])

    with m1:

        st.markdown(
            f"### Report `{selected_report_id}`"
        )

        st.write(
            f"**Damage:** {selected_row.get('Prediction', 'Unknown')}"
        )

        st.write(
            f"**Risk:** {selected_row.get('Risk', 'Unknown')}"
        )

        st.write(
            f"**Priority:** {selected_row.get('Priority', 'Medium')}"
        )

        st.write(
            f"**Confidence:** {selected_row.get('Confidence', 'Unknown')}"
        )

        st.write(
            f"**Time:** {selected_row.get('Time', 'Unknown')}"
        )

        st.write(
            f"**Latitude:** {selected_row.get('Latitude', 'N/A')}"
        )

        st.write(
            f"**Longitude:** {selected_row.get('Longitude', 'N/A')}"
        )

        current_status = selected_row.get(
            "Status",
            "Pending"
        )

        current_priority = selected_row.get(
            "Priority",
            suggested_priority(selected_row)
        )

        st.caption(
            "💡 Priority is automatically suggested from the AI result, "
            "but the administrator can change it manually."
        )

        current_notes = str(
            selected_row.get(
                "Admin Notes",
                ""
            )
        )

        new_status = st.selectbox(
            "Update Status",
            [
                "Pending",
                "In Progress",
                "Completed"
            ],
            index=[
                "Pending",
                "In Progress",
                "Completed"
            ].index(current_status)
            if current_status in [
                "Pending",
                "In Progress",
                "Completed"
            ]
            else 0,
            key=f"status_{selected_report_id}"
        )

        new_priority = st.selectbox(
            "Priority",
            [
                "High",
                "Medium",
                "Low"
            ],
            index=[
                "High",
                "Medium",
                "Low"
            ].index(current_priority)
            if current_priority in [
                "High",
                "Medium",
                "Low"
            ]
            else 1,
            key=f"priority_{selected_report_id}"
        )

        new_notes = st.text_area(
            "📝 Admin Notes",
            value=current_notes,
            placeholder="Add maintenance notes or follow-up details...",
            key=f"notes_{selected_report_id}",
            height=120
        )

        if st.button(
            "💾 Save Report Updates",
            type="primary"
        ):

            df.loc[
                selected_index,
                "Status"
            ] = new_status

            df.loc[
                selected_index,
                "Priority"
            ] = new_priority

            df.loc[
                selected_index,
                "Admin Notes"
            ] = new_notes

            df.to_csv(
                REPORT_FILE,
                index=False
            )

            st.success(
                f"✅ Report {selected_report_id} updated successfully."
            )

            st.rerun()

    with m2:

        image_path = os.path.join(
            SAVED_IMAGES_DIR,
            f"{selected_report_id}.jpg"
        )

        if os.path.exists(image_path):

            st.image(
                image_path,
                caption="Saved Damage Image",
                use_container_width=True
            )

        else:

            st.info(
                "No saved image was found for this report."
            )


# ============================================================
# REPORT TABLE
# ============================================================

st.markdown("---")

st.subheader("📋 Reports Table")

display_columns = [
    "Report ID",
    "Image",
    "Prediction",
    "Confidence",
    "Risk",
    "Priority",
    "Status",
    "Admin Notes",
    "Latitude",
    "Longitude",
    "Time"
]

available_columns = [
    column for column in display_columns
    if column in filtered_df.columns
]

st.dataframe(
    filtered_df[available_columns],
    use_container_width=True,
    hide_index=True
)


# ============================================================
# GIS ADMIN MAP
# ============================================================

st.markdown("---")

st.subheader("🗺️ Damage Monitoring Map")

valid_df = filtered_df.copy()

# Completed reports are considered resolved, so they are removed
# from the active damage monitoring map.
if "Status" in valid_df.columns:

    valid_df = valid_df[
        valid_df["Status"].astype(str) != "Completed"
    ]

if (
    "Latitude" in valid_df.columns
    and "Longitude" in valid_df.columns
):

    valid_df = valid_df.dropna(
        subset=[
            "Latitude",
            "Longitude"
        ]
    )

if len(valid_df) > 0:

    first_row = valid_df.iloc[0]

    m = folium.Map(
        location=[
            float(first_row["Latitude"]),
            float(first_row["Longitude"])
        ],
        zoom_start=13
    )

    for _, row in valid_df.iterrows():

        try:

            lat = float(
                row["Latitude"]
            )

            lon = float(
                row["Longitude"]
            )

        except Exception:

            continue

        risk = str(
            row.get(
                "Risk",
                "Unknown"
            )
        )

        if "High" in risk:
            color = "red"

        elif "Medium" in risk:
            color = "orange"

        else:
            color = "green"

        report_id = str(
            row.get(
                "Report ID",
                "Unknown"
            )
        )

        image_path = os.path.join(
            SAVED_IMAGES_DIR,
            f"{report_id}.jpg"
        )

        image_html = ""

        if os.path.exists(
            image_path
        ):

            import base64

            try:

                with open(
                    image_path,
                    "rb"
                ) as image_file:

                    image_base64 = base64.b64encode(
                        image_file.read()
                    ).decode("utf-8")

                image_html = (
                    f'<br><img src="data:image/jpeg;base64,'
                    f'{image_base64}" '
                    f'style="width:250px; '
                    f'max-height:180px; '
                    f'object-fit:contain;">'
                )

            except Exception:
                pass

        popup = (
            f"<div style='width:270px;'>"
            f"<b>Report ID:</b> {report_id}<br>"
            f"<b>Damage:</b> {row.get('Prediction', 'Unknown')}<br>"
            f"<b>Risk:</b> {risk}<br>"
            f"<b>Priority:</b> {row.get('Priority', 'Medium')}<br>"
            f"<b>Status:</b> {row.get('Status', 'Pending')}<br>"
            f"<b>Confidence:</b> {row.get('Confidence', 'Unknown')}<br>"
            f"<b>Time:</b> {row.get('Time', 'Unknown')}"
            f"{image_html}"
            f"</div>"
        )

        folium.CircleMarker(
            location=[
                lat,
                lon
            ],
            radius=9,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            tooltip=report_id,
            popup=folium.Popup(
                popup,
                max_width=320
            )
        ).add_to(m)

    st_folium(
        m,
        width=1100,
        height=550,
        returned_objects=[]
    )

else:

    st.info(
        "No active GPS reports match the current filters. "
        "Completed reports are hidden from the monitoring map."
    )


# ============================================================
# DOWNLOAD
# ============================================================

st.markdown("---")

st.subheader("📥 Export")

csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "📥 Download Filtered Reports",
    data=csv_data,
    file_name="filtered_damage_reports.csv",
    mime="text/csv",
    use_container_width=True
)