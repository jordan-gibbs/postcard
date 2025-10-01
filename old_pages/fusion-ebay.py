import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time
import json  # For safer error handling

# --- Configuration ---
# IMPORTANT: Set this to your deployed FastAPI base URL
BACKEND_API_URL = "https://postcard-worker-9m6z.onrender.com"

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="eBay Postcard Processor",
    page_icon="🖼️",
    layout="wide",
)

# --- Session State Initialization ---
if "submit_success_message" not in st.session_state:
    st.session_state.submit_success_message = None
if "submitted_geo_job_id" not in st.session_state:
    st.session_state.submitted_geo_job_id = None
if "submitted_nongeo_job_id" not in st.session_state:
    st.session_state.submitted_nongeo_job_id = None
if "submitted_single_job_id" not in st.session_state:
    st.session_state.submitted_single_job_id = None
if "links_text_area_value" not in st.session_state:
    st.session_state.links_text_area_value = ""
if "reset_after_submit_at" not in st.session_state:
    st.session_state.reset_after_submit_at = None

# --- Helpers ---

def submit_processing_job(links):
    """
    Calls the router endpoint to submit a mixed list of links.
    The router should split into geo + non-geo and kick off the right flows.
    Accepts both new (two job IDs) and legacy (single job_id) schemas.
    """
    endpoint = f"{BACKEND_API_URL}/router/process-postcards"
    payload = {"links": links}
    response = None
    try:
        response = requests.post(endpoint, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("Backend API timed out. The job may still start — check 'View Results' later.")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to {BACKEND_API_URL}. Is the backend up?")
        return None
    except requests.exceptions.RequestException as e:
        # Try to surface any FastAPI-provided 'detail'
        detail = "N/A"
        if response is not None:
            try:
                detail = response.json().get("detail", response.text)
            except json.JSONDecodeError:
                detail = response.text
        st.error(f"Error submitting job: {e}. Details: {detail}")
        return None


def get_all_jobs():
    """Fetch the list of all jobs (any type) from the backend."""
    endpoint = f"{BACKEND_API_URL}/jobs"
    response = None
    try:
        response = requests.get(endpoint, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("Backend API timed out while fetching job list.")
        return []
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to {BACKEND_API_URL}. Is the backend up?")
        return []
    except requests.exceptions.RequestException as e:
        detail = "N/A"
        if response is not None:
            try:
                detail = response.json().get("detail", response.text)
            except json.JSONDecodeError:
                detail = response.text
        st.error(f"Error fetching job list: {e}. Details: {detail}")
        return []


# --- Pages ---

def submit_job_page():
    st.title("🖼️ Submit New eBay Postcard Job")
    st.write("Paste image URLs (front/back pairs) to start a new job. The backend router will send each pair to the right flow automatically.")

    # --- Timed Reset Logic (keeps the success banner visible briefly) ---
    if st.session_state.reset_after_submit_at is not None:
        remaining = int(st.session_state.reset_after_submit_at - time.time())
        if remaining <= 0:
            # full state reset and rerun
            st.session_state.clear()
            st.rerun()
            return
        else:
            # Persist success UI while the countdown runs
            if st.session_state.submit_success_message:
                st.success(st.session_state.submit_success_message)
            # Show IDs if present
            if st.session_state.submitted_geo_job_id or st.session_state.submitted_nongeo_job_id or st.session_state.submitted_single_job_id:
                with st.expander("Submitted Job IDs", expanded=True):
                    if st.session_state.submitted_geo_job_id:
                        st.write(f"**Geographic Job ID:** `{st.session_state.submitted_geo_job_id}`")
                    if st.session_state.submitted_nongeo_job_id:
                        st.write(f"**Non-Geographic Job ID:** `{st.session_state.submitted_nongeo_job_id}`")
                    if st.session_state.submitted_single_job_id:
                        st.write(f"**Job ID:** `{st.session_state.submitted_single_job_id}`")
                st.info("Go to **View Results** to check status and download.")
            return

    # --- Form ---
    links_input = st.text_area(
        "Paste image URLs (one per line). Must be an even number for front/back pairs.",
        height=220,
        value=st.session_state.links_text_area_value,
        key="links_input_widget",
    )

    col_submit, col_clear = st.columns([1, 1])
    with col_submit:
        if st.button("Submit Job", type="primary"):
            links = [ln.strip() for ln in links_input.splitlines() if ln.strip()]
            if not links:
                st.warning("Please enter some image URLs.")
                return
            if len(links) % 2 != 0:
                st.warning("You have an odd number of links. Please submit complete front/back pairs.")
                return

            with st.spinner("Submitting to backend router..."):
                result = submit_processing_job(links)

            if not result:
                st.session_state.submit_success_message = None
                return

            # Handle both schemas:
            geo_job_id = None
            nongeo_job_id = None
            single_job_id = None

            # New router schema (examples of possible keys)
            for k in ["geographic_job_id", "geo_job_id", "geographic", "geo"]:
                if isinstance(result.get(k), dict) and "job_id" in result[k]:
                    geo_job_id = result[k]["job_id"]
                elif isinstance(result.get(k), str):
                    geo_job_id = result[k]
            for k in ["non_geographic_job_id", "nongeo_job_id", "non_geographic", "nongeo", "nonloc"]:
                if isinstance(result.get(k), dict) and "job_id" in result[k]:
                    nongeo_job_id = result[k]["job_id"]
                elif isinstance(result.get(k), str):
                    nongeo_job_id = result[k]

            # Legacy single endpoint
            if not geo_job_id and not nongeo_job_id and "job_id" in result:
                single_job_id = result["job_id"]

            st.session_state.submitted_geo_job_id = geo_job_id
            st.session_state.submitted_nongeo_job_id = nongeo_job_id
            st.session_state.submitted_single_job_id = single_job_id

            # Success messaging
            if geo_job_id or nongeo_job_id:
                msg_lines = ["Job(s) submitted successfully via router!"]
                if geo_job_id:
                    msg_lines.append(f"- Geographic Job ID: `{geo_job_id}`")
                if nongeo_job_id:
                    msg_lines.append(f"- Non-Geographic Job ID: `{nongeo_job_id}`")
                st.session_state.submit_success_message = "\n".join(msg_lines)
            elif single_job_id:
                st.session_state.submit_success_message = (
                    f"Job submitted successfully! Job ID: `{single_job_id}`"
                )
            else:
                st.session_state.submit_success_message = "Submitted, but no job IDs returned."

            st.session_state.links_text_area_value = ""
            st.session_state.reset_after_submit_at = time.time() + 6
            st.balloons()
            st.rerun()

    with col_clear:
        if st.button("Clear Form"):
            st.session_state.links_text_area_value = ""
            st.experimental_rerun()


def view_results_page():
    st.title("🗃️ eBay Processed Postcards")
    st.write("Find all jobs below. Use the selector to download a finished CSV.")

    st.button("Refresh Results", key="refresh_results_button")

    jobs = get_all_jobs()
    if not jobs:
        st.info("No jobs found yet. Submit a new job or refresh.")
        return

    # Table-friendly dataframe
    df_jobs = pd.DataFrame(jobs)
    # Timestamps are already in ET from the backend; just format for display
    if "timestamp" in df_jobs.columns:
        df_jobs["timestamp"] = pd.to_datetime(df_jobs["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S ET")

    # Reorder/rename
    cols = ["job_id", "status", "timestamp", "filename", "error_message"]
    df_jobs = df_jobs[[c for c in cols if c in df_jobs.columns]]
    df_jobs = df_jobs.rename(columns={
        "job_id": "Job ID",
        "status": "Status",
        "timestamp": "Updated At (ET)",
        "filename": "File Name",
        "error_message": "Error Details",
    })

    # --- Download panel ---
    st.subheader("Download a Result")
    valid_ids = [j["job_id"] for j in jobs if j.get("job_id")]
    if not valid_ids:
        st.info("No downloadable jobs yet.")
    else:
        sel = st.selectbox("Select Job ID", options=valid_ids, key="download_job_id")
        if sel:
            # Find that job’s details
            details = next((j for j in jobs if j.get("job_id") == sel), None)
            if not details:
                st.warning("Could not locate job in listing.")
            else:
                status = details.get("status")
                fname = details.get("filename", f"postcards_job_{sel}.csv")
                if status == "completed":
                    download_endpoint = f"{BACKEND_API_URL}/jobs/{sel}/download"
                    resp = None
                    try:
                        with st.spinner(f"Preparing '{fname}'..."):
                            resp = requests.get(download_endpoint, timeout=60)
                            resp.raise_for_status()
                            csv_bytes = resp.content
                        st.download_button(
                            label=f"Download {fname}",
                            data=csv_bytes,
                            file_name=fname,
                            mime="text/csv",
                            key=f"dl_{sel}",
                        )
                    except requests.exceptions.Timeout:
                        st.error(f"Download timed out for job {sel}.")
                    except requests.exceptions.ConnectionError:
                        st.error(f"Could not connect to backend for download of job {sel}.")
                    except requests.exceptions.RequestException as e:
                        detail = "N/A"
                        if resp is not None:
                            try:
                                detail = resp.json().get("detail", resp.text)
                            except json.JSONDecodeError:
                                detail = resp.text
                        st.error(f"Error fetching file: {e}. Details: {detail}")
                elif status in {"pending", "processing"}:
                    st.info(f"Job `{sel}` is **{status}**. Please refresh later.")
                elif status == "failed":
                    st.error(f"Job `{sel}` **failed**. Error: {details.get('error_message', 'No details provided.')}")
                else:
                    st.warning("Unknown job status.")

    # --- All jobs table ---
    st.subheader("All Jobs")
    st.dataframe(df_jobs, use_container_width=True, hide_index=True)


# --- Main ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Submit Job", "View Results"], key="page_selection")

if page == "Submit Job":
    submit_job_page()
else:
    view_results_page()
