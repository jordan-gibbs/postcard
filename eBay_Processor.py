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
import html


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

    # Reconstruct the title by placing the year back in its original position
    if year:
        final_title = title_without_year[:year_position].strip() + " " + year + " " + title_without_year[year_position:].strip()
        final_title = final_title.strip()
    else:
        final_title = title_without_year.strip()

    final_title_length = len(final_title)  # +1 for the space before the year
    words = final_title.split()

    while final_title_length > 80 and words:
        # Remove a word from the beginning until we are under 80 characters
        words.pop(0)
        final_title = ' '.join(words)
        final_title_length = len(final_title.strip()) + 1

    return final_title


def convert_to_html(text):
    # Escape special HTML characters
    text = html.escape(text)
    # Split text into paragraphs by splitting on double newlines
    paragraphs = text.split('\n\n')
    # Wrap each paragraph in <p> tags, replacing single newlines with <br>
    html_paragraphs = ['<p>{}</p>'.format(paragraph.replace('\n', '<br>')) for paragraph in paragraphs]
    # Join the paragraphs
    return ''.join(html_paragraphs)


def remove_code_from_url(url):
    # Remove the dynamic "xx-001" segment from the URL
    return re.sub(r'(archives/)[^/]+/', r'\1', url)

def save_postcards_to_csv(postcards_details, first_column_set):
    headers = ["front_image_link", "back_image_link", "SKU", "Title", "Cancel Title", "Destination Title", "Region",
               "Country", "City",
               "Era",
               "Description"]
    rows = []

    boilerplate = """Please inspect the scanned postcard image for condition. All cards are sold as is. Payment is due within 3 days of purchase or we may re-list it for other buyers. Please note we offer VOLUME DISCOUNTS (2 for 10%, 3 for 15%, 4 for 20%, and 10+ for 30%) so please check out our massive store selection. We have 1,000s of cards in stock with views from nearly every state and country, all used with messages, stamps, interesting postal routes, and more. Thank you so much for visiting postal*connection, you are appreciated.

PS - WE BUY POSTCARDS! Top prices paid for good collections.
"""

    counter = 1  # Initialize counter for SKU

    for postcard in postcards_details:
        try:
            details = json.loads(postcard["details"])
        except json.JSONDecodeError:
            details = {}  # Handle JSON decoding errors gracefully

        title = details.get("Title", "")
        city = details.get("City", "")
        description = details.get("Description", "").strip()
        total_description = f"{description}\n\n{boilerplate}" if description else boilerplate

        # Convert the total_description to HTML
        total_description_html = convert_to_html(total_description)

        cleaned_title = clean_title(title, city)


        # Stuff title with origin and destination
        origin = details.get("Origin City", "")
        print(origin)
        destination = details.get("Destination City", "")
        print(destination)

        def check_variable(variable_to_check):
            return variable_to_check.lower() in first_column_set

        if check_variable(origin):
            origin = origin
        else:
            origin = ""

        if check_variable(destination):
            destination = destination
        else:
            destination = ""

        if len(cleaned_title) + len(origin) < 80:
            cancel_title = f"{origin} {cleaned_title}"
        else:
            cancel_title = ""

        if len(cleaned_title) + len(destination) < 80:
            destination_title = f"{cleaned_title} {destination}"
        else:
            destination_title = ""


        front_image_link = postcard.get("front_image_link", "")
        back_image_link = postcard.get("back_image_link", "")
        # Define the pattern for 'xx-xxx' (alphanumeric characters with a hyphen)
        pattern = r'[A-Za-z0-9]{2}-[A-Za-z0-9]{3}'
        # Search for the pattern in the front_image_link
        match = re.search(pattern, front_image_link)
        if match:
            sku_prefix = match.group(0)  # If a match is found, use it
            print(f"Matched SKU Prefix: {sku_prefix}")  # For debugging purposes
        else:
            sku_prefix = 'NOSKU'  # Default SKU prefix if no match is found
            print("No SKU Prefix found, using default: NOSKU")  # For debugging purposes
        # Generate SKU
        SKU = f'{sku_prefix}_{counter:02d}'
        # Increment counter
        counter += 1
        # For debugging purposes, print SKU
        print("Generated SKU:", SKU)

        row = {
            "original_index": postcard["original_index"],  # Include original index
            "front_image_link": front_image_link,
            "back_image_link": back_image_link,
            "SKU": SKU,
            "Title": cleaned_title,
            "Cancel Title": cancel_title,
            "Destination Title": destination_title,
            "Region": details.get("Region", ""),
            "Country": details.get("Country", ""),
            "City": details.get("City", ""),
            "Era": details.get("Era", ""),
            "Description": total_description_html  # Use the HTML-formatted description
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
            writer = csv.DictWriter(file, fieldnames=headers, quotechar='"', quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(sorted_rows)

    return tmp_file.name


# Helper functions (you should have these defined elsewhere in your code)
def convert_to_html(text):
    # Implement your HTML conversion logic here
    return text  # Placeholder implementation


def clean_title(title, city):
    # Implement your title cleaning logic here
    return title  # Placeholder implementation


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
    5. **Era**: You must identify the proper era of the card from these choices: Undivided Back (1901-1907), Divided Back (1907-1915), White Border (1915-1930), Linen (1930-1945), Photochrome (1945-now). You can only choose from those.
    6. **Description** Write a short, descriptive, and non-flowery description of the card, preferably including details that aren't necessarily found in the title, containing elements such as (e.g., "written from a mother to a son" "references to farming" "reference to WWI" "a child's handwriting"). You must also definitively state where the card was sent, the recipients name, address, town, and state/country in the description. 
    7. **Destination City** If available, write the destination city, ONLY THE CITY. 
    8. **Origin City** If available, write the origin city, ONLY THE CITY.


    Please output the result in the following structure, and NOTHING else:

    Example Output:
    {
        "Title": "Vintage Georgia Postcard SAVANNAH Beach Highway Palms Oleanders 1983",
        "Region": "Georgia",
        "Country": "USA",
        "City": "Savannah",
        "Era": "Photochrome (1945-now)",
        "Description": "Features a scenic highway lined with palms and oleanders, likely promoting beach tourism. 
        Sent postmarked from Savannah, with a brief note about a family road trip.",
        "Destination City": "Tallahassee",
        "Origin City": "Aaron"
    }

    Another Example:
    {
        "Title": "Antique Wyoming Postcard YELLOWSTONE National Park Gibbon Falls Haynes 1913",
        "Region": "Wyoming",
        "Country": "USA",
        "City": "Yellowstone",
        "Description": "Shows Gibbon Falls, part of a Haynes collection of early park photography. Likely from a 
        traveler describing natural wonders, with references to early park infrastructure.",
        "Destination City": "Billings",
        "Origin City": "Cheyenne"
    }

    Another Example:
    {
        "Title": "Antique Florida Postcard ST. PETERSBURG John's Pass Bridge Fishing 1957",
        "Region": "Florida",
        "Country": "USA",
        "City": "St. Petersburg",
        "Era": "Divided Back (1907-1915)",
        "Description": "Depicts fishermen at John's Pass Bridge, a popular tourist and fishing spot. Postcard 
        mentions a family vacation, with references to warm weather and abundant fishing.",
        "Destination City": "Boston",
        "Origin City": "Chicago"
    }

    Another Example:
    {
        "Title": "Vintage Virginia Postcard NEWPORT NEWS Mariner's Museum Cover to Milwaukee Post 1999",
        "Region": "Virginia",
        "Country": "USA",
        "City": "Newport News",
        "Era": "Photochrome (1945-now)",
        "Description": "Features a museum display, likely sent from a visitor to Newport News. Includes mention of 
        shipbuilding history, with a personal note about travel to Milwaukee.",
        "Destination City": "Billings",
        "Origin City": "Newport News"
    }

    Another Example:
    {
        "Title": "Vintage Tennessee Postcard MEMPHIS Romeo & Juliet in Cotton Field Black 1938",
        "Region": "Tennessee",
        "Country": "USA",
        "City": "Memphis",
        "Era": "Linen (1930-1945)",
        "Description": "Displays a staged romantic scene of two figures in a cotton field. Likely includes commentary on Southern agriculture or nostalgia, with dated cultural imagery.",
        "Destination City": "Bozeman",
        "Origin City": "Memphis"
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
        "model": "gpt-4o-2024-08-06",
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

    import pandas as pd

    df = pd.read_csv('us-post-offices.csv', usecols=[0])
    first_column = df.iloc[:, 0].str.lower()
    first_column_set = set(first_column)

    st.title("🖼️eBay Processor")
    st.write("Upload a set of postcard image links (front and back) for Ebay processing.")

    api_key = os.getenv("OPENAI_API_KEY")
    links_input = st.text_area("Paste image URLs (one per line)")
    links = [link for link in links_input.splitlines() if link.strip()] if links_input else []

    if links:
        if "csv_data1" not in st.session_state:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Download and save images from the links
                with st.spinner("Getting image files..."):
                    postcards = download_images_from_links(links, tmp_dir)

                with st.spinner("Processing images..."):
                    # Process postcards using workers
                    postcards_details = process_postcards_in_folder(api_key, postcards, workers=10)
                    st.write("Processing complete!")

                    # Save the results to a CSV file
                    csv_file = save_postcards_to_csv(postcards_details, first_column_set)

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
