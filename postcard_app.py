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

# Function to save postcards details to a CSV file
def save_postcards_to_csv(postcards_details):
    headers = ["front_image_link", "back_image_link", "Title", "Region", "Country", "City"]
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
        row = {
            "front_image_link": postcard["front_image_link"],  # Use exact image link for front
            "back_image_link": postcard["back_image_link"],  # Use exact image link for back
            "Title": cleaned_title,
            "Region": details.get("Region", ""),
            "Country": details.get("Country", ""),
            "City": details.get("City", "")
        }
        rows.append(row)

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
        page_title="Postcard Processor",
        page_icon="🖼",  # You can choose any emoji as the icon
        layout="centered",
    )

    st.title("🖼️Postcard Processor")
    st.write("Upload a set of postcard image links (front and back) for processing.")

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
