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
    # Handle empty title or city cases
    if not title:
        return title

    # Remove unwanted quotation marks
    title = title.replace('"', '').replace("'", "")

    # Split title into words and decapitalize where needed
    words = title.split()
    decapitalized_title = []

    for word in words:
        if word.isupper() and len(word) > 2:  # Decapitalize words that are all uppercase
            decapitalized_word = word.lower().capitalize()
            decapitalized_title.append(decapitalized_word)
        else:
            decapitalized_title.append(word)

    # Join words back into the title
    title = ' '.join(decapitalized_title)

    # Ensure non-empty city for the pattern matching
    if city:
        city_pattern = re.compile(r'\b' + re.escape(city) + r'\b', re.IGNORECASE)
        if city_pattern.search(title):
            title = city_pattern.sub(city.upper(), title)  # Replace city name with uppercase

    # Now let's handle ensuring the string length doesn't exceed 80 characters

    # Find the 4-digit year or a year with 's' (like 1950s)
    year_pattern = re.compile(r'(\b\d{4}s?\b)')
    year_match = year_pattern.search(title)

    year = ''
    year_position = -1
    if year_match:
        year = year_match.group(0)  # Extract the year or decade (e.g., 1999 or 1950s)
        year_position = year_match.start()  # Get the position of the year in the string

    # If year is found, split the title into parts: before and after the year
    if year:
        before_year = title[:year_position].strip()  # Title before the year
        after_year = title[year_position + len(year):].strip()  # Title after the year (if any)
        title_without_year = before_year + " " + after_year  # Combine both parts without year
    else:
        title_without_year = title  # If no year is found, treat the whole title normally

    # Check if the title length (including year) exceeds 80 characters
    full_title_length = len(title_without_year.strip()) + len(year.strip()) + 1  # +1 for the space before the year
    words = title_without_year.split()

    while full_title_length > 80 and words:
        # Remove a word from the beginning until we are under 80 characters
        words.pop(0)
        title_without_year = ' '.join(words)
        full_title_length = len(title_without_year.strip()) + len(year.strip()) + 1

    # Reconstruct the title by placing the year back in its original position
    if year:
        final_title = title_without_year[:year_position].strip() + " " + year + " " + title_without_year[year_position:].strip()
        final_title = final_title.strip()
    else:
        final_title = title_without_year.strip()

    return final_title

def save_postcards_to_csv(postcards_details):
    headers = ["front_image_link", "back_image_link", "Title", "Region", "Country", "City"]
    rows = []

    for postcard in postcards_details:
        try:
            details = json.loads(postcard["details"])
        except json.JSONDecodeError:
            details = {}  # Handle JSON decoding errors gracefully

        title = details.get("Title", "")
        city = details.get("City", "")
        cleaned_title = clean_title(title, city)
        row = {
            "original_index": postcard["original_index"],  # Include original index
            "front_image_link": postcard["front_image_link"],
            "back_image_link": postcard["back_image_link"],
            "Title": cleaned_title,
            "Region": details.get("Region", ""),
            "Country": details.get("Country", ""),
            "City": details.get("City", "")
        }
        rows.append(row)

    # Sort rows by the original index
    sorted_rows = sorted(rows, key=lambda x: x["original_index"])

    # Remove 'original_index' before writing to CSV
    for row in sorted_rows:
        row.pop("original_index", None)

    # Write the sorted data to the CSV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        with open(tmp_file.name, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(sorted_rows)

    return tmp_file.name


def save_postcards_to_final_csv(postcards_details):
    headers = ["front_image_link", "back_image_link", "Title", "Region", "Country", "City"]
    rows = []

    for postcard in postcards_details:
        try:
            details = json.loads(postcard["details"])
        except json.JSONDecodeError:
            details = {}  # Handle JSON decoding errors gracefully

        title = details.get("Title", "")
        city = details.get("City", "")
        cleaned_title = clean_title(title, city)
        row = {
            "original_index": postcard["original_index"],  # Include original index
            "front_image_link": postcard["front_image_link"],
            "back_image_link": postcard["back_image_link"],
            "Title": cleaned_title,
            "Region": details.get("Region", ""),
            "Country": details.get("Country", ""),
            "City": details.get("City", "")
        }
        rows.append(row)

    # Sort rows by the original index
    sorted_rows = sorted(rows, key=lambda x: x["original_index"])

    # Remove 'original_index' before writing to CSV
    for row in sorted_rows:
        row.pop("original_index", None)

    # Write the sorted data to the CSV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        with open(tmp_file.name, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(sorted_rows)

    return tmp_file.name



# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def download_images_from_links(links, tmp_dir):
    postcards = []
    # Process links in pairs (front and back)
    for idx in range(0, len(links), 2):
        try:
            # Download front image
            front_link = links[idx]
            front_response = requests.get(front_link)
            front_response.raise_for_status()
            front_image_filename = f"image_{idx:03d}.jpg"
            front_image_path = os.path.join(tmp_dir, front_image_filename)
            with open(front_image_path, "wb") as f:
                f.write(front_response.content)

            # Download back image if available
            if idx + 1 < len(links):
                back_link = links[idx + 1]
                back_response = requests.get(back_link)
                back_response.raise_for_status()
                back_image_filename = f"image_{idx + 1:03d}.jpg"
                back_image_path = os.path.join(tmp_dir, back_image_filename)
                with open(back_image_path, "wb") as f:
                    f.write(back_response.content)
            else:
                back_link = None
                back_image_filename = None
                back_image_path = None

            postcards.append({
                "original_index": idx // 2,
                "front_image_filename": front_image_filename,
                "back_image_filename": back_image_filename,
                "front_image_path": front_image_path,
                "back_image_path": back_image_path,
                "front_image_link": front_link,
                "back_image_link": back_link
            })
        except Exception as e:
            print(f"Failed to download images at index {idx}: {e}")
    return postcards



# Function to get postcard details using the API
def get_postcard_details(api_key, front_image_path, back_image_path):
    front_image_base64 = encode_image(front_image_path)
    back_image_base64 = encode_image(back_image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    prompt = """
    You are given two images of a vintage or antique postcard:

    1. The first image is the **front** of the postcard.
    2. The second image is the **back** of the postcard, which contains text and possibly other relevant details.

    I need you to analyze both the front and back images and provide the following information:

    1. **Title**: Create a descriptive title for the postcard based on the front and back. The title should be **90 characters or less**.
    2. **Region**: Identify the U.S. state or region mentioned in the postcard.
    3. **Country**: Identify the country mentioned on the postcard.
    4. **City**: Identify the city or major landmark mentioned on the postcard.

    Please output the result in the following structure, and NOTHING else:

    Example Output:
    {
        "Title": "Vintage Georgia Postcard SAVANNAH Beach Highway Palms Oleanders 1983",
        "Region": "Georgia",
        "Country": "USA",
        "City": "Savannah"
    }

    Another Example:
    {
        "Title": "Antique Wyoming Postcard YELLOWSTONE National Park Gibbon Falls Haynes 1913",
        "Region": "Wyoming",
        "Country": "USA",
        "City": "Yellowstone"
    }

    Another Example:
    {
        "Title": "Antique Florida Postcard ST. PETERSBURG John's Pass Bridge Fishing 1957",
        "Region": "Florida",
        "Country": "USA",
        "City": "St. Petersburg"
    }

    Another Example:
    {
        "Title": "Vintage Virginia Postcard NEWPORT NEWS Mariner's Museum Cover to Milwaukee Post 1999",
        "Region": "Virginia",
        "Country": "USA",
        "City": "Newport News"
    }

    Another Example:
    {
        "Title": "Vintage Tennessee Postcard MEMPHIS Romeo & Juliet in Cotton Field Black 1938",
        "Region": "Tennessee",
        "Country": "USA",
        "City": "Memphis"
    }

    If any of the information cannot be found on the postcard, please output just '' for that field.

    Always try to put the year in if available. 

    Never ever shorten a city name, ie never do New York -> NY. 

    Always put the city in all caps in the title field, i.e. 'BOSTON' but never put it in all caps in the City Field.  
    Never put the attraction itself in all caps, ONLY the city. 

    Never output any commas within the title.

    Never output any sort of formatting block, i.e. ```json just output the raw string.

    Try to max out the 90 character limit in the title field, keyword stuff if you must. 

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


def process_batch(api_key, batch):
    postcards_details = []
    for postcard in batch:
        front_image_path = postcard["front_image_path"]
        back_image_path = postcard["back_image_path"]
        front_image_link = postcard["front_image_link"]
        back_image_link = postcard["back_image_link"]
        original_index = postcard["original_index"]

        postcard_details = get_postcard_details(api_key, front_image_path, back_image_path)
        postcards_details.append({
            "original_index": original_index,
            "front_image": postcard["front_image_filename"],
            "back_image": postcard["back_image_filename"],
            "front_image_link": front_image_link,
            "back_image_link": back_image_link,
            "details": postcard_details
        })
    return postcards_details



def get_image_index(filename):
    m = re.search(r'image_(\d+)\.jpg', filename)
    return int(m.group(1)) if m else -1


def process_postcards_in_folder(api_key, postcards, workers=10):
    total_postcards = len(postcards)
    if total_postcards == 0:
        raise ValueError("No postcards to process.")

    # Split postcards into batches
    batch_size = max(1, total_postcards // workers)
    postcard_batches = [postcards[i:i + batch_size] for i in range(0, total_postcards, batch_size)]

    postcards_details = []
    failed_batches = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_batch, api_key, batch): idx
            for idx, batch in enumerate(postcard_batches)
        }
        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                result = future.result()
                postcards_details.extend(result)
            except Exception as exc:
                print(f"Batch {batch_idx} generated an exception: {exc}")
                failed_batches.append(batch_idx)

    # Re-run failed batches if any
    if failed_batches:
        for batch_idx in failed_batches:
            batch = postcard_batches[batch_idx]
            try:
                result = process_batch(api_key, batch)
                postcards_details.extend(result)
            except Exception as exc:
                print(f"Batch {batch_idx} failed again: {exc}")

    # Sort postcards by original index
    postcards_details = sorted(postcards_details, key=lambda x: x["original_index"])

    return postcards_details



def main():
    st.set_page_config(
        page_title="eBay Processor",
        page_icon="🖼",  # You can choose any emoji as the icon
        layout="centered",
    )

    st.title("🖼️eBay Processor")
    st.write("Upload a set of postcard image links (front and back) for Ebay processing.")

    api_key = os.getenv("OPENAI_API_KEY")
    links_input = st.text_area("Paste image URLs (one per line)")
    links = [link for link in links_input.splitlines() if link.strip()] if links_input else []

    if links:
        if "csv_data" not in st.session_state:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Download and save images from the links
                with st.spinner("Getting image files..."):
                    postcards = download_images_from_links(links, tmp_dir)

                with st.spinner("Processing images..."):
                    # Process postcards using workers
                    postcards_details = process_postcards_in_folder(api_key, postcards, workers=10)
                    st.write("Processing complete!")

                    # Save the results to a CSV file
                    csv_file = save_postcards_to_csv(postcards_details)

                    # Read the CSV file and store data in session state
                    with open(csv_file, "rb") as f:
                        st.session_state.csv_data1 = f.read()

        # Create the download button, using stored CSV data
        st.download_button(
            label="Download CSV",
            data=st.session_state.csv_data1,
            file_name="postcards.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
