import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time

# --- Configuration ---
# IMPORTANT: Replace this with the actual URL of your backend service on Render.com
# Example: "https://your-service-name.onrender.com"
BACKEND_API_URL = "http://localhost:8000"  # Use localhost for local testing, change for deployment

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="eBay Postcard Processor",
    page_icon="🖼️",
    layout="wide",  # Use wide layout for better table display
)

# --- Session State Initialization ---
if "current_page" not in st.session_state:
    st.session_state.current_page = "Submit Job"
if "submit_success_message" not in st.session_state:
    st.session_state.submit_success_message = None
if "submitted_job_id" not in st.session_state:
    st.session_state.submitted_job_id = None


# --- Helper Functions ---

def submit_processing_job(links):
    """Calls the backend API to submit a new processing job."""
    endpoint = f"{BACKEND_API_URL}/process-postcards"
    payload = {"links": links}
    try:
        response = requests.post(endpoint, json=payload, timeout=30)  # Increased timeout for API call
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.Timeout:
        st.error(
            f"Backend API timed out after 30 seconds. The job might still be running. Please check 'View Results' page later.")
        return None
    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not connect to the backend API at {BACKEND_API_URL}. Please ensure the backend service is running.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(
            f"Error submitting job to backend: {e}. Details: {response.json().get('detail', 'N/A') if response else 'N/A'}")
        return None


def get_all_jobs():
    """Fetches a list of all processed jobs from the backend."""
    endpoint = f"{BACKEND_API_URL}/jobs"
    try:
        response = requests.get(endpoint, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("Backend API timed out while fetching job list.")
        return []
    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not connect to the backend API at {BACKEND_API_URL}. Please ensure the backend service is running.")
        return []
    except requests.exceptions.RequestException as e:
        st.error(
            f"Error fetching job list from backend: {e}. Details: {response.json().get('detail', 'N/A') if response else 'N/A'}")
        return []


# --- Streamlit Pages ---

def submit_job_page():
    st.title("🖼️ Submit New Postcard Processing Job")
    st.write("Paste image URLs (front and back pairs) to start a new processing job.")

    links_input = st.text_area("Paste image URLs (one per line)", height=200, key="links_input")

    if st.button("Submit Job", type="primary"):
        links = [link.strip() for link in links_input.splitlines() if link.strip()]
        if not links:
            st.warning("Please enter some image URLs.")
            return

        # Check if links count is even
        if len(links) % 2 != 0:
            st.warning("Please ensure you have an even number of links (front and back image pairs).")
            return

        with st.spinner("Submitting job to backend..."):
            result = submit_processing_job(links)

        if result and result.get("job_id"):
            st.session_state.submit_success_message = (
                f"Job submitted successfully! Your Job ID is: `{result['job_id']}`. "
                "You can view its status and download the result on the 'View Results' page."
            )
            st.session_state.submitted_job_id = result['job_id']
            # Reset the textarea input after successful submission
            st.session_state.links_input = ""
            st.success(st.session_state.submit_success_message)
            st.balloons()  # Visual confirmation
        else:
            st.error("Failed to submit job. Please check the logs or try again.")

    # Display success message persistently after redirect if it was set
    if st.session_state.submit_success_message:
        st.success(st.session_state.submit_success_message)
        # Clear it after displaying to prevent it from showing on subsequent refreshes
        # st.session_state.submit_success_message = None
        # For this design, let's keep it until the user clears it or submits a new job.


def view_results_page():
    st.title("🗃️ Processed Postcard Results")
    st.write("Here you can see all past processing jobs and download their results.")

    if st.button("Refresh Results"):
        st.cache_data.clear()  # Clear cache to fetch latest data

    jobs = get_all_jobs()

    if not jobs:
        st.info("No jobs found or an error occurred while fetching. Try refreshing or submitting a new job.")
        return

    # Convert job data to a pandas DataFrame for easy display
    df_jobs = pd.DataFrame(jobs)

    # Convert timestamp to a readable format
    df_jobs['timestamp'] = pd.to_datetime(df_jobs['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S UTC')

    # Reorder columns for better presentation
    df_jobs = df_jobs[['job_id', 'status', 'timestamp', 'filename', 'error_message']]
    df_jobs = df_jobs.rename(columns={
        'job_id': 'Job ID',
        'status': 'Status',
        'timestamp': 'Completed At (UTC)',
        'filename': 'File Name',
        'error_message': 'Error Details'
    })

    st.dataframe(df_jobs, use_container_width=True, hide_index=True)

    st.subheader("Download Specific Result")
    # Provide a selectbox or input for job ID to download
    job_ids = [job['job_id'] for job in jobs]
    selected_job_id = st.selectbox("Select Job ID to Download", options=job_ids, key="download_job_id")

    if selected_job_id:
        # Fetch the full job details to get the filename and check status
        # This is a bit inefficient as we already have partial data, but robust.
        selected_job_details = next((job for job in jobs if job['job_id'] == selected_job_id), None)

        if selected_job_details and selected_job_details.get("status") == "completed":
            download_endpoint = f"{BACKEND_API_URL}/jobs/{selected_job_id}/download"
            filename = selected_job_details.get("filename", f"postcards_job_{selected_job_id}.csv")

            try:
                # Use a direct requests.get for download_button's data parameter
                # The data parameter expects bytes, so fetch it directly
                response = requests.get(download_endpoint, stream=True, timeout=60)  # Increased timeout for download
                response.raise_for_status()
                csv_bytes = response.content  # Read content into memory for download button

                st.download_button(
                    label=f"Download {filename}",
                    data=csv_bytes,
                    file_name=filename,
                    mime="text/csv",
                    key=f"download_{selected_job_id}"
                )
            except requests.exceptions.Timeout:
                st.error(f"Download request timed out for job {selected_job_id}.")
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to backend for download of job {selected_job_id}.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching file for job {selected_job_id}: {e}")

        elif selected_job_details and selected_job_details.get("status") in ["pending", "processing"]:
            st.info(f"Job `{selected_job_id}` is currently **{selected_job_details['status']}**. Please refresh later.")
        elif selected_job_details and selected_job_details.get("status") == "failed":
            st.error(
                f"Job `{selected_job_id}` **failed**. Error: {selected_job_details.get('error_message', 'No details.')}")
        else:
            st.warning("Select a job to see its status or download.")


# --- Main App Logic (Sidebar Navigation) ---

st.sidebar.title("Navigation")
page_selection = st.sidebar.radio(
    "Go to",
    ["Submit Job", "View Results"],
    key="page_selection"
)

# Render the selected page
if page_selection == "Submit Job":
    submit_job_page()
elif page_selection == "View Results":
    view_results_page()