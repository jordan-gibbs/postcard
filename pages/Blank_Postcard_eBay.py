import os
import zipfile
import tempfile
import streamlit as st
import base64
import requests
import csv
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import re
import json
import numpy as np
import html
import time
import logging
import threading
import unicodedata
import pandas as pd

# Configure logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# Ensure logs go to the console in deployed environments:
if "RENDER" in os.environ:
    logging.getLogger().addHandler(logging.StreamHandler())

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

    # Check if initial title length is 75 or more
    if final_title_length >= 71:
        words = final_title.split()

        # Step 1: Remove "Vintage" and "Postcard" if they exist in the list
        words = [word for word in words if word not in ["Vintage", "Postcard", "Antique"]]
        final_title = ' '.join(words)
        final_title_length = len(final_title.strip())

        # Step 2: If still over 71, keep removing the first word until the title is under 71 characters
        while final_title_length > 71 and words:
            words.pop(0)
            final_title = ' '.join(words)
            final_title_length = len(final_title.strip())

    logging.debug(f"Cleaned title: {final_title}")
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

def save_postcards_to_csv(postcards_details, first_column_set, all_rows, decade):
    logging.info("Starting save_postcards_to_csv")
    headers = ["front_image_link", "SKU", "Title", "Destination Title", "Combo Title", "Region",
               "Country", "City",
               "Era",
               "Description"]

    boilerplate = """Please inspect the scanned postcard image for condition. All cards are sold as is. Payment is due within 3 days of purchase or we may re-list it for other buyers. Please note we offer VOLUME DISCOUNTS (2 for 10%, 3 for 15%, 4 for 20%, and 10+ for 30%) so please check out our massive store selection. We have 1,000s of cards in stock with views from nearly every state and country, all used with messages, stamps, interesting postal routes, and more. Thank you so much for visiting postal*connection, you are appreciated.
     Use code with caution.
    PS - WE BUY POSTCARDS! Top prices paid for good collections.
    """

    counter = 1  # Initialize counter for SKU

    for postcard in postcards_details:
        logging.debug(f"Processing postcard with original index: {postcard.get('original_index')}")

        try:
            details = json.loads(postcard["details"])
        except json.JSONDecodeError:
            logging.error(
                f"JSONDecodeError for postcard with original index: {postcard.get('original_index')}. Details: {postcard.get('details')}")
            details = {"Title": "", "Region": "", "Country": "", "City": "", "Era": "", "Description": "",
                       "Origin City": "", "Destination City": ""}  # Set default values

        title = details.get("Title", "")
        city = details.get("City", "")
        description = details.get("Description", "").strip()
        total_description = f"{description}\n\n{boilerplate}" if description else boilerplate

        # Convert the total_description to HTML
        total_description_html = convert_to_html(total_description)

        cleaned_title = clean_title(title, city)

        # Stuff title with origin and destination
        origin = details.get("Origin City", "")
        logging.debug(f"Origin: {origin}")
        destination = details.get("Destination City", "")
        logging.debug(f"Destination: {destination}")

        def check_variable(variable_to_check):
            return variable_to_check.lower() in first_column_set

        def validate_destination(destination):
            if not destination:
                return ""
            # Use regex to split destination into location and state code parts
            match = re.match(r"^(.*?)(?=\s[A-Z]{2}$)", destination)

            if match:
                location = match.group(1).strip()  # Extract the location part
                logging.debug(f"Location from Destination: {location}")
                state_code = destination[len(location):].strip()  # Extract the state code part
                logging.debug(f"State code from Destination: {state_code}")
                if check_variable(location):  # Run the validity check on the location
                    # Reconstruct and return if valid
                    return f"{location.title()} {state_code}"
            return ""  # Return an empty string if invalid

        if check_variable(origin):
            origin = origin
        else:
            origin = ""

        destination = validate_destination(destination)
        logging.debug(f"Validated Destination: {destination}")

        if origin.lower() != city.lower():
            # Step 1: Generate the full titles
            cancel_title = f"{origin} {cleaned_title}"
        else:
            cancel_title = ""

        if destination != "":
            destination_title = f"{cleaned_title} to {destination}"
        else:
            destination_title = ""

        # Step 2: Truncate cancel_title if it exceeds 75 characters
        if len(cancel_title) >= 75:
            cancel_words = cancel_title.split()

            # Remove "Vintage" and "Postcard" if present
            cancel_words = [word for word in cancel_words if word not in ["Vintage", "Postcard", "Antique"]]

            # Remove duplicate words past the first mention, case-insensitively
            seen_words = set()
            unique_cancel_words = []
            for word in cancel_words:
                lower_word = word.lower()  # Convert word to lowercase for comparison
                if lower_word not in seen_words:
                    unique_cancel_words.append(word)  # Keep the original casing
                    seen_words.add(lower_word)

            # Truncate from the beginning until the title is under 75 characters
            while len(' '.join(unique_cancel_words)) >= 75 and unique_cancel_words:
                unique_cancel_words.pop(0)

            cancel_title = ' '.join(unique_cancel_words) if len(' '.join(unique_cancel_words)) < 75 else ""

        # Step 3: Truncate destination_title if it exceeds 75 characters
        if len(destination_title) >= 75:
            destination_words = destination_title.split()
            # Remove "Vintage" and "Postcard" if present
            destination_words = [word for word in destination_words if word not in ["Vintage", "Postcard", "Antique"]]
            # Truncate from the beginning until the title is under 75 characters
            while len(' '.join(destination_words)) >= 75 and destination_words:
                destination_words.pop(0)
            destination_title = ' '.join(destination_words) if len(' '.join(destination_words)) < 75 else ""

        if cancel_title != "":
            cleaned_title = cancel_title

        # Results
        logging.debug(f"Cancel Title: {cancel_title}")
        logging.debug(f"Destination Title: {destination_title}")

        front_image_link = postcard.get("front_image_link", "")
        # Define the pattern for 'xx-xxx' (alphanumeric characters with a hyphen)
        pattern = r'[A-Za-z0-9]{2}-[A-Za-z0-9]{3}'
        # Search for the pattern in the front_image_link
        match = re.search(pattern, front_image_link)
        if match:
            sku_prefix = match.group(0)  # If a match is found, use it
            logging.debug(f"Matched SKU Prefix: {sku_prefix}")  # For debugging purposes
        else:
            sku_prefix = 'NOSKU'  # Default SKU prefix if no match is found
            logging.warning("No SKU Prefix found, using default: NOSKU")  # For debugging purposes
        # Generate SKU
        # SKU = f'{sku_prefix}_{counter:02d}'
        SKU = f'{sku_prefix}'
        # Increment counter
        counter += 1
        # For debugging purposes, print SKU
        # print("Generated SKU:", SKU)

        combo_title = destination_title if destination_title else cleaned_title

        row = {
            # "original_index": postcard["original_index"],  # Include original index
            "front_image_link": front_image_link,
            "SKU": SKU,
            "Title": f"{cleaned_title} {decade}",
            # "Cancel Title": cancel_title,
            # "Destination Title": destination_title,
            # "Combo Title": combo_title,
            "Region": details.get("Region", ""),
            "Country": details.get("Country", ""),
            "City": details.get("City", ""),
            # "Era": details.get("Era", ""),
            "Description": total_description_html  # Use the HTML-formatted description
        }
        all_rows.append(row)

    print(all_rows)


    logging.info(f"save_postcards_to_csv completed.")
    return all_rows

def clean_text(text):
    if pd.isnull(text):
        return ''

    # Define a dictionary of common alternate characters to replace
    replacements = {
        '‘': "'", '’': "'", '“': '"', '”': '"', '–': '-', '—': '-',
        '…': '...', '«': '"', '»': '"', '‹': "'", '›': "'"
    }

    # Normalize the text to decompose accents
    text = unicodedata.normalize('NFKD', str(text))


    # Decode HTML entities like & to their actual characters
    text = html.unescape(text)

    # Replace any alternate characters with their standard ASCII equivalents
    for alt_char, standard_char in replacements.items():
        text = text.replace(alt_char, standard_char)

    # Remove any non-Latin characters
    text = ''.join([c for c in text if unicodedata.category(c) != 'Mn'])

    # Remove unwanted symbols, keeping basic punctuation and alphanumeric characters
    text = re.sub(r'[^A-Za-z0-9\s.,?!\'"()%<>-_/]', '', text)

    return text

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logging.error(f"Error encoding image: {image_path}. Error: {e}")
        return None

def download_images_from_links(links, tmp_dir):
    logging.info("Starting download_images_from_links")
    postcards = []
    total_links = len(links)
    progress_text = "Downloading images. Please wait."
    my_bar = st.progress(0, text=progress_text)

    # Lock for thread-safe updates
    progress_lock = threading.Lock()
    postcards_lock = threading.Lock()
    downloaded_count = 0

    def download_pair(idx):
        nonlocal downloaded_count
        try:
            front_link = links[idx]
            logging.debug(f"Downloading front image from: {front_link}")
            front_response = requests.get(front_link)
            front_response.raise_for_status()
            front_image_filename = f"image_{idx:03d}.jpg"
            front_image_path = os.path.join(tmp_dir, front_image_filename)
            with open(front_image_path, "wb") as f:
                f.write(front_response.content)


            with postcards_lock:
                postcards.append({
                    "original_index": idx,
                    "front_image_filename": front_image_filename,
                    "front_image_path": front_image_path,
                    "front_image_link": front_link,
                })
            logging.debug(f"Finished downloading images for index {idx}")

        except Exception as e:
            logging.error(f"Failed to download images at index {idx}: {e}")
            with postcards_lock:
                postcards.append({
                    "original_index": idx,
                    "front_image_filename": "",
                    "front_image_path": "",
                    "front_image_link": front_link,
                })
        finally:
            with progress_lock:
                downloaded_count += 1
                progress_percentage = downloaded_count / total_links
                my_bar.progress(progress_percentage, text=f"{progress_text} ({int(progress_percentage * 100)}%)")
    # Use ThreadPoolExecutor to download images concurrently
    max_workers = min(20, (len(links) + 1) // 2)  # Adjust the number of workers as needed
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx in range(0, total_links, 2):
            executor.submit(download_pair, idx)

    # Ensure progress bar is complete
    my_bar.progress(1.0, text="Download complete!")
    time.sleep(1)
    my_bar.empty()
    logging.info("Finished download_images_from_links")

    return postcards

def get_postcard_details(api_key, front_image_path, timeout=20, max_workers=100):
    """Get postcard details with timeout and parallel processing."""
    logging.debug(f"Starting get_postcard_details for front image: {front_image_path}")

    def api_call():
        return _get_postcard_details_helper(api_key, front_image_path)  # Removed back_image_path

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future = executor.submit(api_call)  # Submit the API call to the thread pool
        try:
            result = future.result(timeout=timeout)  # Wait with a timeout
            logging.debug(f"get_postcard_details completed for front image: {front_image_path}")
            return result
        except concurrent.futures.TimeoutError:
            logging.warning(
                f"Timeout occurred in get_postcard_details, skipping call for front image: {front_image_path}")
            return '{"Title": "", "Region": "", "Country": "", "City": "", "Era": "", "Description": ""}'
        except Exception as e:
            logging.error(
                f"Exception in get_postcard_details, skipping call for front image: {front_image_path}. Error: {e}")
            return '{"Title": "", "Region": "", "Country": "", "City": "", "Era": "", "Description": ""}'

def _get_postcard_details_helper(api_key, front_image_path):
    """Helper function to do the actual API call."""
    front_image_base64 = encode_image(front_image_path)

    if front_image_base64 is None:
        logging.error(
            f"Could not encode one of the images for API call in _get_postcard_details_helper. Front: {front_image_path}")
        return '{"Title": "", "Region": "", "Country": "", "City": "", "Era": "", "Description": ""}'

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    # ... [Your prompt logic here]...
    prompt = """
    You are given an image of the front of a vintage or antique postcard. You must focus on the imagery on the card.
    
    I need you to analyze the image and provide the following information based on the imagery and the wording on the card:

    1. **Title**: Create a descriptive title for the postcard based on the image. The title should be **80 characters or less**.
    2. **Region**: Identify the state or region mentioned in the postcard.
    3. **Country**: Identify the country mentioned on the postcard.
    4. **City**: Identify the city or major landmark mentioned on the postcard.
    5. **Description** Write a short, descriptive, and non-flowery description of the card, You must also definitively state the intricacies of the card's content. Do not mention any information about it being blank or not filled out. 


    Please output the result in the following structure, and NOTHING else:

    Example Outputs:
    {
        "Title": "Vintage Georgia Postcard SAVANNAH Beach Highway",
        "shortTitle": "Vintage Georgia Postcard SAVANNAH Beach",
        "Region": "Georgia",
        "Country": "USA",
        "City": "Savannah",
        "Description": "Features a scenic highway lined with tall palm trees, vibrant oleanders, and a glimpse of the ocean horizon in the distance. The bright summer sun glistens on the blacktop, while vintage automobiles pass by, capturing the bustling energy of coastal tourism."
    }
    
    Example 2:
    {
        "Title": "Antique Wyoming Postcard YELLOWSTONE National Park Gibbon Falls",
        "shortTitle": "Antique Wyoming Postcard YELLOWSTONE Gibbon Falls",
        "Region": "Wyoming",
        "Country": "USA",
        "City": "Yellowstone",
        "Description": "A dramatic depiction of Gibbon Falls from the iconic Haynes collection of early park photography. The cascading waters tumble over layered rock formations, framed by the lush pine forest."
    }
    
    Example 3:
    {
        "Title": "Antique Florida Postcard ST. PETERSBURG John's Pass Bridge",
        "shortTitle": "Antique Florida Postcard ST. PETERSBURG John's Pass",
        "Region": "Florida",
        "Country": "USA",
        "City": "St. Petersburg",
        "Description": "Depicts fishermen at John's Pass Bridge, a lively scene with colorful fishing boats docked along the waterfront. The bright Florida sun illuminates the sparkling ocean waters, showcasing the vibrant energy of this bustling seaside landmark."
    }
    
    Example 4:
    {
        "Title": "Vintage Virginia Postcard NEWPORT NEWS Mariner's Museum",
        "shortTitle": "Vintage Virginia Postcard NEWPORT NEWS",
        "Region": "Virginia",
        "Country": "USA",
        "City": "Newport News",
        "Description": "Showcases a detailed maritime exhibit featuring historic ship models and nautical artifacts. Dramatic lighting highlights the intricate craftsmanship, evoking a sense of maritime heritage and adventure."
    }
    
    Example 5:
    {
        "Title": "Vintage Tennessee Postcard MEMPHIS Romeo & Juliet in Cotton Field",
        "shortTitle": "Vintage Tennessee Postcard MEMPHIS Cotton Field",
        "Region": "Tennessee",
        "Country": "USA",
        "City": "Memphis",
        "Description": "Displays a staged romantic scene featuring two figures costumed as Romeo and Juliet, set against a sprawling cotton field in full bloom. The pastel hues and stylized poses reflect a nostalgic, theatrical take on Southern agriculture."
    }

    If any of the information cannot be found on the postcard, please output just '' for that field.

    Always try to put the year in if available. 

    Never ever shorten a city name, ie never do New York -> NY. 

    Always put the city in all caps in the title field, i.e. 'BOSTON' but never put it in all caps in the City Field.  
    Never put the attraction itself in all caps, ONLY the city. 

    Never output any commas within the title.

    Never output any sort of formatting block, i.e. ```json just output the raw string.

    Try to max out the 80 character limit in the title field, keyword stuff if you must. Never repeat the city or any words within the title ever. 
    
    Try to stuff as much information into the title as possible, but no years or decades. 
    
    
    The short title can be creatively made, using same formatting guidelines, just make sure it is 1-2 words shorter than the actual first title you wrote.
    Make sure to carefully analyze the **text on the back** of the postcard as well, since it may contain valuable information like the city, region, or country.
    """

    payload = {
        "model": "gpt-4.1",
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
                    }
                ]
            }
        ],
        "max_tokens": 300,
        # "type": "json_object"
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        response_json = response.json()
        details = response_json['choices'][0]['message']['content'] if 'choices' in response_json else "Details not available"
        logging.debug(f"API Response: {details}")
        return details
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return '{"Title": "", "Region": "", "Country": "", "City": "", "Era": "", "Description": ""}'
    except Exception as e:
        logging.error(f"Exception in _get_postcard_details_helper, skipping call. Error: {e}")
        return '{"Title": "", "Region": "", "Country": "", "City": "", "Era": "", "Description": ""}'

def get_secondary_postcard_details(api_key, front_image_path, back_image_path, timeout=20, max_workers=100):
    """Get secondary postcard details with timeout and parallel processing."""
    logging.debug(f"Starting get_secondary_postcard_details for front image: {front_image_path}")

    def api_call():
        return _get_secondary_postcard_details_helper(api_key, front_image_path, back_image_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future = executor.submit(api_call)  # Submit the API call to the thread pool
        try:
            result = future.result(timeout=timeout)  # Wait with a timeout
            logging.debug(f"get_secondary_postcard_details completed for front image: {front_image_path}")
            return result
        except concurrent.futures.TimeoutError:
            logging.warning(
                f"Timeout occurred in get_secondary_postcard_details, skipping call for front image: {front_image_path}")
            return '{"Destination City": "", "Origin City": ""}'
        except Exception as e:
            logging.error(
                f"Exception in get_secondary_postcard_details, skipping call for front image: {front_image_path}. Error: {e}")
            return '{"Destination City": "", "Origin City": ""}'

def _get_secondary_postcard_details_helper(api_key, front_image_path, back_image_path):
    """Helper function to do the actual API call."""

    front_image_base64 = encode_image(front_image_path)
    back_image_base64 = encode_image(back_image_path)

    if front_image_base64 is None or back_image_base64 is None:
        logging.error(
            f"Could not encode one of the images for API call in _get_secondary_postcard_details_helper. Front: {front_image_path}, Back: {back_image_path}")
        return '{"Destination City": "", "Origin City": ""}'

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    prompt = """
    You are given two images of a vintage or antique postcard:

    1. The first image is the **front** of the postcard.
    2. The second image is the **back** of the postcard, which contains text and possibly other relevant details.

    I need you to analyze both the front and back images and provide the following information:

    1. **Origin City** If available, write the origin city, ONLY THE CITY. This is found within the circular black postage stamp.
    This is known as the cancel of the card. This is often written in circular text around the stamp. 

    2. **Destination City** If available, write the destination city and the state, no comma eg Billings MT. This will likely be written on the card in handwriting. 

    Please output the result in the following structure, and NOTHING else:

    Example Output:
    {
        "Destination City": "Tallahassee TN",
        "Origin City": "Aaron"
    }

    Another Example:
    {
        "Destination City": "Billings MT",
        "Origin City": "Cheyenne"
    }

    Another Example:
    {
        "Destination City": "Boston IL",
        "Origin City": "Chicago"
    }

    Another Example:
    {
        "Destination City": "Billings MT",
        "Origin City": "Newport News"
    }

    Another Example:
    {
        "Destination City": "Bozeman MT",
        "Origin City": "Memphis"
    }

    If any of the information cannot be found on the postcard, please output just '' for that field.

    YOU MUST USE THE STATE SHORTCODE FOR THE DESTINATION CITY, I.E., MT, IL, HI, ETC. IF YOU OUTPUT THE ACTUAL FULL STATE NAME, YOU HAVE FAILED YOUR TASK.

    Never ever shorten a city name, ie never do New York -> NY. 

    Never output any sort of formatting block, i.e. ```json just output the raw string.

    Make sure to carefully analyze the **text on the back** of the postcard as well, since it may contain valuable information.
    """

    payload = {
        "model": "gpt-4.1",
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
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        response_json = response.json()
        details = response_json['choices'][0]['message'][
            'content'] if 'choices' in response_json else "Details not available"
        logging.debug(f"Secondary API response: {details}")
        return details
    except requests.exceptions.RequestException as e:
        logging.error(f"Secondary API request failed: {e}")
        return '{"Destination City": "", "Origin City": ""}'
    except Exception as e:
        logging.error(f"Exception in _get_secondary_postcard_details_helper, skipping call. Error: {e}")
        return '{"Destination City": "", "Origin City": ""}'


def process_batch(api_key, batch):
    logging.info("Starting process_batch")
    postcards_details = []
    for postcard in batch:
        front_image_path = postcard["front_image_path"]
        front_image_link = postcard["front_image_link"]
        original_index = postcard["original_index"]
        logging.debug(f"Processing postcard from original index {original_index} in process_details")

        postcard_details = get_postcard_details(api_key, front_image_path)

        try:
            # Convert JSON strings to dictionaries
            postcard_details = json.loads(postcard_details)

        except json.JSONDecodeError as e:
            logging.error(
                f"JSONDecodeError in process_batch for postcard at {original_index}: {e}. Details: {postcard_details}")
            postcard_details = {"Title": "", "Region": "", "Country": "", "City": "", "Era": "", "Description": ""}


        # Convert the merged dictionary back to a JSON string
        postcard_details = json.dumps(postcard_details)

        postcards_details.append({
            "original_index": original_index,
            "front_image": postcard["front_image_filename"],
            "front_image_link": front_image_link,
            "details": postcard_details
        })
    logging.info("Finished process_batch")
    return postcards_details


def get_image_index(filename):
    m = re.search(r'image_(\d+).jpg', filename)
    return int(m.group(1)) if m else -1


def process_postcards_in_folder(api_key, postcards, workers=200):
    logging.info("Starting process_postcards_in_folder")
    total_postcards = len(postcards)
    if total_postcards == 0:
        logging.error("No postcards to process.")
        raise ValueError("No postcards to process.")

    # Initialize progress bar
    progress_text = "Processing postcards. Please wait."
    my_bar = st.progress(0, text=progress_text)

    # Split postcards into batches
    batch_size = max(1, total_postcards // workers)
    postcard_batches = [postcards[i:i + batch_size] for i in range(0, total_postcards, batch_size)]

    postcards_details = []
    failed_batches = []

    total_batches = len(postcard_batches)
    completed_batches = 0

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
                logging.error(f"Batch {batch_idx} generated an exception: {exc}")
                failed_batches.append(batch_idx)
            finally:
                # Update progress bar for each completed batch
                completed_batches += 1
                progress_value = completed_batches / total_batches  # Normalize to 0.0 - 1.0
                my_bar.progress(progress_value, text=f"{progress_text} ({int(progress_value * 100)}%)")

    # Re-run failed batches if any
    if failed_batches:
        logging.warning(f"Retrying failed batches: {failed_batches}")
        for batch_idx in failed_batches:
            batch = postcard_batches[batch_idx]
            try:
                result = process_batch(api_key, batch)
                postcards_details.extend(result)
                # Update progress for retried batches
                completed_batches += 1
                progress_value = completed_batches / total_batches  # Normalize to 0.0 - 1.0
                my_bar.progress(progress_value, text=f"{progress_text} ({int(progress_value * 100)}%)")
            except Exception as exc:
                logging.error(f"Batch {batch_idx} failed again: {exc}")

    # Mark progress as complete
    my_bar.progress(1.0, text="Processing complete!")
    time.sleep(1)
    my_bar.empty()

    # Sort postcards by original index
    postcards_details = sorted(postcards_details, key=lambda x: x["original_index"])
    logging.info("Finished process_postcards_in_folder")

    return postcards_details

def download_callback():
    st.session_state.downloaded = True
    #st.rerun()


def main():
    if "links" not in st.session_state:
        st.session_state.links = None

    if "all_csv_data" not in st.session_state:
        st.session_state.all_csv_data = None

    if "downloaded" not in st.session_state:
        st.session_state.downloaded = False


    st.set_page_config(
        page_title="eBay Processor",
        page_icon="🖼",  # You can choose any emoji as the icon
        layout="centered",
    )

    import pandas as pd

    df = pd.read_csv('us-post-offices.csv', usecols=[0])
    first_column = df.iloc[:, 0].str.lower()
    first_column_set = set(first_column)

    st.title("🖼️Blank eBay Processor")
    st.write("Upload a set of postcard image links (front and back) for Ebay processing.")

    decades = ["1900s-10s", "1920s-30s", "1940s-50s", "1960s-70s", "1980s-90s", "2000s-10s"]

    decade = st.selectbox(
        "Select a decade to append to the end of each title:",
        options=decades,)

    api_key = os.getenv("OPENAI_API_KEY")

    if st.session_state.downloaded:
        st.write("Data has been downloaded.")
        return  # Stop here if the download happened

    links_input = st.text_area("Paste image URLs (one per line)")

    if links_input:
        links = [link for link in links_input.splitlines() if link.strip()] if links_input else []
        st.session_state.links = links
    else:
        pass

    if st.session_state.links:
        links = st.session_state.links
        batch_size = 50  # Set batch size
        all_rows = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            for i in range(0, len(links), batch_size):
                logging.info(f"Starting processing for batch: {i // batch_size + 1}")
                current_links = links[i:i + batch_size]
                st.write(
                    f"Processing batch {i // batch_size + 1} of {math.ceil(len(links) / batch_size)}")  # Showing the batch currently being processed.

                # Download and save images from the links
                with st.spinner(f"Getting image files for batch {i // batch_size + 1}..."):
                    logging.info(f"Downloading images for batch: {i // batch_size + 1}")
                    postcards = download_images_from_links(current_links, tmp_dir)

                with st.spinner(f"Processing images for batch {i // batch_size + 1}..."):
                    # Process postcards using workers
                    logging.info(f"Processing images for batch: {i // batch_size + 1}")
                    postcards_details = process_postcards_in_folder(api_key, postcards, workers=100)
                    st.write("Processing complete!")

                    # Save the results to a CSV file
                    logging.info(f"Saving CSV for batch: {i // batch_size + 1}")
                    all_rows = save_postcards_to_csv(postcards_details, first_column_set, all_rows, decade)

        if all_rows:
            # create a pandas dataframe
            df = pd.DataFrame(all_rows)

            # Create a copy of the original DataFrame to store the cleaned data
            df_cleaned = df.copy()

            columns_to_clean = [col for i, col in enumerate(df_cleaned.columns[2:8]) if
                                i + 3 != 6]  # 3:10 - description, title, dest title, combo, region, country, city and era are here
            df_cleaned.loc[:, columns_to_clean] = df_cleaned.loc[:, columns_to_clean].applymap(
                clean_text)
            df_cleaned = df_cleaned.fillna('')

            # Save the cleaned data to a new CSV file
            logging.info(f"Saving cleaned CSV file.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                df_cleaned.to_csv(tmp_file.name, index=False)

                # Read the CSV file and store data in session state
                with open(tmp_file.name, "rb") as f:
                    st.session_state.all_csv_data = f.read()

                if st.session_state.all_csv_data:
                    # Create the download button, using stored CSV data
                    st.download_button(
                        label=f"Download Final CSV",
                        data=st.session_state.all_csv_data,
                        file_name=f"postcards_final.csv",
                        mime="text/csv",
                        on_click=download_callback
                    )


if __name__ == "__main__":
    main()