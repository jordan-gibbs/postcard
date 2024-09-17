import os
import zipfile
import tempfile
import streamlit as st
import base64
import requests
import csv
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import json
import numpy as np


def clean_title(title, city):
    # nothing
    return title


def save_postcards_to_csv(postcards_details):
    # Updated headers to include the new structure
    headers = ["front_image_link", "back_image_link", "Title", "Date", "Region", "State", "Country", "City", "Destination City", "Recipient", "Year", "Description"]
    rows = []

    # Collect all the rows in a list first
    for postcard in postcards_details:
        try:
            details = json.loads(postcard["details"])
        except json.JSONDecodeError:
            details = {}  # Handle JSON decoding errors gracefully

        title = details.get("Title", "")
        city = details.get("City", "")
        cleaned_title = clean_title(title, city)

        # Create a row with all the fields, including the new ones
        row = {
            "front_image_link": postcard.get("front_image_link", ""),
            "back_image_link": postcard.get("back_image_link", ""),
            "Title": cleaned_title,
            "Date": details.get("Date", ""),
            "State": details.get("State", ""),
            "Country": details.get("Country", ""),
            "City": details.get("City", ""),
            "Destination City": details.get("Destination City", ""),
            "Recipient": details.get("Recipient", ""),
            "Year": details.get("Year", ""),
            "Description": details.get("Description", ""),
        }
        rows.append(row)

    # Sort the rows by the "front_image_link" in ascending alphabetical order (for sequential order)
    sorted_rows = sorted(rows, key=lambda x: x["front_image_link"])

    # Write the sorted data to the CSV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        with open(tmp_file.name, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(sorted_rows)

    return tmp_file.name

    # Sort the rows by the "front_image_link" in ascending alphabetical order (for sequential order)
    sorted_rows = sorted(rows, key=lambda x: x["front_image_link"])

    # Write the sorted data to the CSV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        with open(tmp_file.name, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(sorted_rows)

    return tmp_file.name


# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def download_images_from_links(links, tmp_dir):
    image_files = []
    image_links = []
    for idx, link in enumerate(links):
        try:
            response = requests.get(link)
            response.raise_for_status()
            image_path = os.path.join(tmp_dir, f"image_{idx}.jpg")
            with open(image_path, "wb") as f:
                f.write(response.content)
            image_files.append(f"image_{idx}.jpg")
            image_links.append(link)  # Save the exact link
        except Exception as e:
            print(f"Failed to download image from {link}: {e}")
        print(image_files)
        print(image_links)
    return image_files, image_links


# Function to get postcard details using the API
def get_postcard_details(api_key, front_image_path, back_image_path):
    front_image_base64 = encode_image(front_image_path)
    back_image_base64 = encode_image(back_image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    prompt = """
    You are a postcard analyzer.
    You are given two images of a vintage or antique postcard:

    1. The first image is the **front** of the postcard.
    2. The second image is the **back** of the postcard, which contains text and possibly other relevant details.

    I need you to analyze both the front and back images and provide the following information and output the result in 
    the following structure, and NOTHING else:

    Example Output:
    {
        "Title": "University of South Carolina Columbia - March 6, 1908 - Vintage Postcard",        
        "Date": "March 6"
        "City": "Columbia"
        "State": "South Carolina",
        "Country": "USA",
        "Destination City": "Walhalla SC"
        "Recipient": "Gertrude Smith"
        "Year": 1908
        "Description": "This vintage postcard, dated March 6, 1908, features Science Hall at the University of South Carolina in Columbia, South Carolina. The front displays an artistic rendering of the impressive classical architecture of the building, surrounded by lush greenery. The back of the postcard, addressed to Gertrude Smith in Walhalla, South Carolina, includes a brief handwritten message from the sender, discussing personal affairs and expressing a longing for quiet, alongside a green one-cent stamp and a postmark from Columbia, SC."
    }

    Another Example:
    {
        "Title": "Boston—Minot Ledge Lighthouse - July 29, 1909 - Vintage Postcard",
        "Date": "July 29, 1909",
        "City": "Boston"
        "State": "Massachusetts",
        "Country": "USA",
        "Destination City": "Billings MT"
        "Recipient": "Dresden A Smith"
        "Year": 1909
        "Description": "This vintage postcard, dated July 29, 1909, depicts the Minot Ledge Lighthouse in Boston, Massachusetts. The front features a beautifully rendered image of the lighthouse under moonlight, casting a serene reflection over the surrounding water. The back of the postcard contains a handwritten message addressing Mr. Dresden A. Smith in Billings, Montana. The sender briefly mentions their visit to Boston, noting they only had a few hours and didn't get a chance to shop for gifts."
    }
    
        Another Example:
    {
        "Title": "Columbia South Carolina—Main Street - July 3, 1908 - Vintage Postcard",
        "Date": "July 3"
        "City": "Columbia"
        "State": "South Carolina",
        "Country": "USA",
        "Destination City": "Walhalla SC"
        "Recipient": "DA Smith"
        "Year": 1908
        "Description": "This vintage postcard, postmarked July 3, 1908, showcases a bustling view of Main Street looking from the Capitol in Columbia, South Carolina. The image captures a moment in time with early 20th-century architecture lining the street, pedestrians visible on the sidewalks, and a clear view down the busy thoroughfare. Sent to DA Smith in Walhalla, South Carolina, the postcard features a green one-cent stamp and is a charming artifact from the period, providing a glimpse into the everyday life and urban landscape of Columbia at the time."
    }
    
    If any of the information cannot be found on the postcard, please output just "" for that field. Never fake 
    anything or try to guess. 

    Always try to put the year in if available. 

    Never ever shorten a city name, ie never do New York -> NY. 

    Never output any sort of formatting block, i.e. ```json just output the raw string.
    
    Never discuss any missing information in the description, jsut write it as if there is nothing missing. 

    Make sure to carefully analyze the **text on the back** of the postcard as well, since it may contain valuable information like the city, region, or country.
    """

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{front_image_base64}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{back_image_base64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300,
        # "type": "json_object"
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_json = response.json()
    details = response_json['choices'][0]['message'][
        'content'] if 'choices' in response_json else "Details not available"
    return details


# Function for a worker to process its batch of images
def process_batch(api_key, folder_path, image_files, image_links):
    postcards_details = []
    # Iterate over the images in pairs
    for i in range(0, len(image_files), 2):
        if i + 1 < len(image_files) and i + 1 < len(image_links):  # Ensure there's a corresponding pair of links
            front_image_path = os.path.join(folder_path, image_files[i])
            back_image_path = os.path.join(folder_path, image_files[i + 1])

            # Ensure the correct image links are used
            front_image_link = image_links[i]  # Get the corresponding front image link
            back_image_link = image_links[i + 1]  # Get the corresponding back image link

            postcard_details = get_postcard_details(api_key, front_image_path, back_image_path)
            postcards_details.append({
                "front_image": image_files[i],
                "back_image": image_files[i + 1],
                "front_image_link": front_image_link,  # Store the front image link
                "back_image_link": back_image_link,  # Store the back image link
                "details": postcard_details
            })
            print(postcards_details)  # Optional: for debugging
    return postcards_details


# Main function to distribute tasks to workers in parallel
def process_postcards_in_folder(api_key, folder_path, image_links, workers=10):
    # List and sort all image files in the folder
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith((".jpg", ".jpeg", ".png"))])

    total_files = len(image_files)

    # Split image files and image links into batches for each worker
    batches_files = np.array_split(image_files, workers)
    batches_links = np.array_split(image_links, workers)  # Split image_links to match the batches of image_files

    postcards_details = []
    failed_batches = []  # To store failed batches for reprocessing

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit batches of image files and corresponding image links to the process_batch function
        futures = {
            executor.submit(process_batch, api_key, folder_path, batch_files, batch_links): i
            for i, (batch_files, batch_links) in enumerate(zip(batches_files, batches_links))
        }
        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                result = future.result()
                postcards_details.extend(result)
            except Exception as exc:
                print(f"Worker {worker_id + 1} generated an exception: {exc}")
                failed_batches.append(worker_id)  # Keep track of failed batches

    # Re-run the failed batches if any
    if failed_batches:
        for worker_id in failed_batches:
            batch_files = batches_files[worker_id]
            batch_links = batches_links[worker_id]
            try:
                result = process_batch(api_key, folder_path, batch_files, batch_links)
                postcards_details.extend(result)
            except Exception as exc:
                print(f"Worker {worker_id + 1} failed again: {exc}")

    return postcards_details


def main():
    st.set_page_config(
        page_title="Website Processor",
        page_icon="🖼",  # You can choose any emoji as the icon
        layout="centered",
    )

    st.title("🌐Website Processor")
    st.write("Upload a set of postcard image links (front and back) to get details for PaleoGreetings.")

    api_key = os.getenv("OPENAI_API_KEY")
    links_input = st.text_area("Paste image URLs (one per line)")
    links = [link for link in links_input.splitlines() if link.strip()] if links_input else []

    if api_key and links:
        if "csv_data" not in st.session_state:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Download and save images from the links
                with st.spinner("Getting image files..."):
                    image_files, image_links = download_images_from_links(links, tmp_dir)

                with st.spinner("Processing images..."):
                    # Process images using 10 workers
                    postcards_details = process_postcards_in_folder(api_key, tmp_dir, image_links, workers=10)
                    print(postcards_details)

                    st.write("Processing complete!")

                    # Save the results to a CSV file
                    csv_file = save_postcards_to_csv(postcards_details)

                    # Read the CSV file and store data in session state
                    with open(csv_file, "rb") as f:
                        st.session_state.csv_data = f.read()

        # Create the download button, using stored CSV data
        st.download_button(
            label="Download CSV",
            data=st.session_state.csv_data,
            file_name="postcards.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
