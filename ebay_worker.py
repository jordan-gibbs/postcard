import os
import zipfile
import tempfile
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
from ebay_formatter import reformat_for_ebay  # Make sure this file is present

# --- MongoDB Imports ---
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from datetime import datetime
from bson import ObjectId

# --- Timezone Imports ---
import pytz  # Added for timezone conversion

# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi.responses import FileResponse, StreamingResponse
import io

from pydantic import BaseModel, Field
from typing import Literal

from google import genai
from google.genai import types
import logging
from PIL import Image
import os
import json





# Define the allowed postcard eras for strict validation
PostcardEra = Literal[
    "Undivided Back (1901-1907)",
    "Divided Back (1907-1915)",
    "White Border (1915-1930)",
    "Linen (1930-1945)",
    "Photochrome (1945-now)",
]

# Create the Pydantic model for the response schema
class PostcardDetails(BaseModel):
    """Defines the schema for the postcard details JSON output."""
    Title: str = Field(description="Descriptive title <= 65 chars. City in caps. Add RPPC if real photo.", max_length=65)
    shortTitle: str = Field(description="A slightly shorter version of the title.")
    Region: str = Field(description="U.S. state or region mentioned.")
    Country: str = Field(description="Country mentioned on the postcard.")
    City: str = Field(description="City or major landmark mentioned.")
    Era: Optional[PostcardEra] = Field(description="The postcard era from the provided list.")
    Description: str = Field(description="Short description including sender/recipient details.")


class SecondaryPostcardDetails(BaseModel):
    """Defines the schema for the origin/destination JSON output."""
    Destination_City: str = Field(alias="Destination City", description="Destination city and state code, e.g., 'Billings MT'.")
    Origin_City: str = Field(alias="Origin City", description="Origin city from the postmark.")

    class Config:
        populate_by_name = True # Important for allowing aliases


# Define a default response for error cases
DEFAULT_DETAILS_RESPONSE = PostcardDetails(
    Title="",
    shortTitle="",
    Region="",
    Country="",
    City="",
    Era=None, # Use None instead of ""
    Description="Error processing postcard details."
).model_dump_json()


# Configure logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# Ensure logs go to the console in deployed environments:
if "RENDER" in os.environ or os.getenv("DEPLOYMENT_ENV") == "production":
    logging.getLogger().addHandler(logging.StreamHandler())

# --- MongoDB Configuration ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "postcard_processing")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "processed_files")

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# --- OpenAI API Key ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY environment variable not set.")

# --- Global variable for us-post-offices.csv data ---
first_column_set = set()
try:
    df_post_offices = pd.read_csv('us-post-offices.csv', usecols=[0])
    first_column_set = set(df_post_offices.iloc[:, 0].str.lower())
    logging.info("Successfully loaded us-post-offices.csv")
except FileNotFoundError:
    logging.error("us-post-offices.csv not found. 'Origin City' validation might be affected.")
except Exception as e:
    logging.error(f"Error loading us-post-offices.csv: {e}")


# --- Helper Functions (unchanged from previous step, except logging/Streamlit related removal) ---

def clean_title(title, city):
    if not title:
        return title
    title = title.replace('"', '').replace("'", "")
    words = title.split()
    decapitalized_title = []
    for word in words:
        if word.isupper() and len(word) > 2:
            decapitalized_word = word.lower().capitalize()
            decapitalized_title.append(decapitalized_word)
        else:
            decapitalized_title.append(word)
    title = ' '.join(decapitalized_title)
    if city:
        city_pattern = re.compile(r'\b' + re.escape(city) + r'\b', re.IGNORECASE)
        if city_pattern.search(title):
            title = city_pattern.sub(city.upper(), title)
    year_pattern = re.compile(r'(\b\d{4}s?\b)')
    year_match = year_pattern.search(title)
    year = ''
    year_position = -1
    if year_match:
        year = year_match.group(0)
        year_position = year_match.start()
    if year:
        before_year = title[:year_position].strip()
        after_year = title[year_position + len(year):].strip()
        title_without_year = before_year + " " + after_year
    else:
        title_without_year = title
    if year:
        final_title = title_without_year[:year_position].strip() + " " + year + " " + title_without_year[
                                                                                      year_position:].strip()
        final_title = final_title.strip()
    else:
        final_title = title_without_year.strip()
    final_title_length = len(final_title)
    words = final_title.split()
    if final_title_length >= 80:
        words = final_title.split()
        words = [word for word in words if word not in ["Vintage", "Postcard", "Antique"]]
        final_title = ' '.join(words)
        final_title_length = len(final_title.strip())
        while final_title_length > 80 and words:
            words.pop(0)
            final_title = ' '.join(words)
            final_title_length = len(final_title.strip())
    logging.debug(f"Cleaned title: {final_title}")
    return final_title


def convert_to_html(text):
    text = html.escape(text)
    paragraphs = text.split('\n\n')
    html_paragraphs = ['<p>{}</p>'.format(paragraph.replace('\n', '<br>')) for paragraph in paragraphs]
    return ''.join(html_paragraphs)


def remove_code_from_url(url):
    return re.sub(r'(archives/)[^/]+/', r'\1', url)


def save_postcards_to_csv(postcards_details, all_rows):
    logging.info("Starting save_postcards_to_csv")
    headers = ["front_image_link", "back_image_link", "SKU", "Title", "Destination Title", "Combo Title", "Region",
               "Country", "City",
               "Era",
               "Description"]
    boilerplate = """Please inspect the scanned postcard image for condition. All cards are sold as is. Payment is due within 3 days of purchase or we may re-list it for other buyers. Please note we offer VOLUME DISCOUNTS (2 for 10%, 3 for 15%, 4 for 20%, and 10+ for 30%) so please check out our massive store selection. We have 1,000s of cards in stock with views from nearly every state and country, all used with messages, stamps, interesting postal routes, and more. Thank you so much for visiting postal*connection, you are appreciated.
     Use code with caution.
    PS - WE BUY POSTCARDS! Top prices paid for good collections.
    """
    for postcard in postcards_details:
        logging.debug(f"Processing postcard with original index: {postcard.get('original_index')}")
        try:
            details = json.loads(postcard["details"])
        except json.JSONDecodeError:
            logging.error(
                f"JSONDecodeError for postcard with original index: {postcard.get('original_index')}. Details: {postcard.get('details')}")
            details = {"Title": "", "Region": "", "Country": "", "City": "", "Era": "", "Description": "",
                       "Origin City": "", "Destination City": ""}
        title = details.get("Title", "")
        city = details.get("City", "")
        description = details.get("Description", "").strip()
        total_description = f"{description}\n\n{boilerplate}" if description else boilerplate
        total_description_html = convert_to_html(total_description)
        cleaned_title = clean_title(title, city)
        origin = details.get("Origin City", "")
        logging.debug(f"Origin: {origin}")
        destination = details.get("Destination City", "")
        logging.debug(f"Destination: {destination}")

        def check_variable(variable_to_check):
            return variable_to_check.lower() in first_column_set

        def validate_destination(destination):
            if not destination:
                return ""
            match = re.match(r"^(.*?)(?=\s[A-Z]{2}$)", destination)
            if match:
                location = match.group(1).strip()
                logging.debug(f"Location from Destination: {location}")
                state_code = destination[len(location):].strip()
                logging.debug(f"State code from Destination: {state_code}")
                if check_variable(location):
                    return f"{location.title()} {state_code}"
            return ""

        if check_variable(origin):
            origin = origin
        else:
            origin = ""
        destination = validate_destination(destination)
        logging.debug(f"Validated Destination: {destination}")
        if origin and origin.lower() != city.lower():
            cancel_title = f"{origin} {cleaned_title}"
        else:
            cancel_title = ""
        if destination != "":
            destination_title = f"{cleaned_title} to {destination}"
        else:
            destination_title = ""
        if len(cancel_title) >= 80:
            cancel_words = cancel_title.split()
            cancel_words = [word for word in cancel_words if word not in ["Vintage", "Postcard", "Antique"]]
            seen_words = set()
            unique_cancel_words = []
            for word in cancel_words:
                lower_word = word.lower()
                if lower_word not in seen_words:
                    unique_cancel_words.append(word)
                    seen_words.add(lower_word)
            while len(' '.join(unique_cancel_words)) >= 80 and unique_cancel_words:
                unique_cancel_words.pop(0)
            cancel_title = ' '.join(unique_cancel_words) if len(' '.join(unique_cancel_words)) < 80 else ""
        if len(destination_title) >= 80:
            destination_words = destination_title.split()
            destination_words = [word for word in destination_words if word not in ["Vintage", "Postcard", "Antique"]]
            while len(' '.join(destination_words)) >= 80 and destination_words:
                destination_words.pop(0)
            destination_title = ' '.join(destination_words) if len(' '.join(destination_words)) < 80 else ""
        if cancel_title != "":
            cleaned_title = cancel_title
        logging.debug(f"Cancel Title: {cancel_title}")
        logging.debug(f"Destination Title: {destination_title}")
        front_image_link = postcard.get("front_image_link", "")
        back_image_link = postcard.get("back_image_link", "")
        try:
            after_archives = front_image_link.split("/archives/", 1)[1]
            SKU = after_archives.split("/", 1)[0]
        except IndexError:
            SKU = "NOSKU"
        combo_title = destination_title if destination_title else cleaned_title
        row = {
            "front_image_link": front_image_link,
            "back_image_link": back_image_link,
            "SKU": SKU,
            "Title": cleaned_title,
            "Destination Title": destination_title,
            "Combo Title": combo_title,
            "Region": details.get("Region", ""),
            "Country": details.get("Country", ""),
            "City": details.get("City", ""),
            "Era": details.get("Era", ""),
            "Description": total_description_html
        }
        all_rows.append(row)
    logging.info(f"save_postcards_to_csv completed.")
    return all_rows


def clean_text(text):
    if pd.isnull(text):
        return ''
    replacements = {
        '‘': "'", '’': "'", '“': '"', '”': '"', '–': '-', '—': '-',
        '…': '...', '«': '"', '»': '"', '‹': "'", '›': "'"
    }
    text = unicodedata.normalize('NFKD', str(text))
    text = html.unescape(text)
    for alt_char, standard_char in replacements.items():
        text = text.replace(alt_char, standard_char)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^A-Za-z0-9\s.,?!\'"()%<>-_/]', '', text)
    return text


def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logging.error(f"Error encoding image: {image_path}. Error: {e}")
        return None



def _get_postcard_details_gemini(front_image_path: str, back_image_path: str) -> str:
    """
    Analyzes front and back postcard images using the Gemini SDK and returns details
    in a structured JSON format.

    NOTE: This function assumes the GOOGLE_API_KEY environment variable is set.

    Args:
        front_image_path: The file path to the front image of the postcard.
        back_image_path: The file path to the back image of the postcard.

    Returns:
        A JSON string containing the extracted postcard details.
    """
    try:
        # The genai.Client() will automatically use the GOOGLE_API_KEY environment variable


        with Image.open(front_image_path) as img_front, Image.open(back_image_path) as img_back:



            # Your original detailed prompt is perfect for guiding the model
            prompt = """
            You are given two images of a vintage or antique postcard:
        
            1. The first image is the **front** of the postcard.
            2. The second image is the **back** of the postcard, which contains text and possibly other relevant details.
        
            I need you to analyze both the front and back images and provide the following information:
        
            1. **Title**: Create a descriptive title for the postcard based on the front and back. The title should be **65 
            characters or less**. If the front of the card has a REAL PHOTOGRAPH (not an illustration or print) in it, 
            be sure to add 'RPPC' at the end of the title. Getting the year in there is very important as well. IT IS NOT AN RPPC if you see 
            things like: fine tonal gradation, dot structure typical of photomechanical printing, if it's a collotype or a 
            photo-type print, it is not an RPPC. Also, if it has a publisher name on the back, it is almost never a real 
            photo. Finally RPPC often have visible gloss, but not always.  
            2. **Region**: Identify the U.S. state or region mentioned in the postcard.
            3. **Country**: Identify the country mentioned on the postcard.
            4. **City**: Identify the city or major landmark mentioned on the postcard.
            5. **Era**: You must identify the proper era of the card from these choices: Undivided Back (1901-1907), Divided Back (1907-1915), White Border (1915-1930), Linen (1930-1945), Photochrome (1945-now). You can only choose from those.
            6. **Description** Write a short, descriptive, and non-flowery description of the card, preferably including details that aren't necessarily found in the title, containing elements such as (e.g., "written from a mother to a son" "references to farming" "reference to WWI" "a child's handwriting"). You must also definitively state where the card was sent, the recipients name, address, town, and state/country in the description. 
        
        
            Please output the result in the following structure, and NOTHING else:
        
            Example Output:
            {
                "Title": "Vintage Georgia Postcard SAVANNAH Beach Highway 1983 RPPC",
                "shortTitle": "Vintage Georgia Postcard SAVANNAH Beach 1983 RPPC",
                "Region": "Georgia",
                "Country": "USA",
                "City": "Savannah",
                "Era": "Photochrome (1945-now)",
                "Description": "Features a scenic highway lined with palms and oleanders, likely promoting beach tourism. 
                Sent postmarked from Savannah, with a brief note about a family road trip."
            }
        
            Another Example:
            {
                "Title": "Antique Wyoming Postcard YELLOWSTONE National Park Gibbon Falls 1913",
                "shortTitle": "Antique Wyoming Postcard YELLOWSTONE Gibbon Falls 1913",
                "Region": "Wyoming",
                "Country": "USA",
                "City": "Yellowstone",
                "Description": "Shows Gibbon Falls, part of a Haynes collection of early park photography. Likely from a 
                traveler describing natural wonders, with references to early park infrastructure."
            }
        
            Another Example:
            {
                "Title": "Antique Florida Postcard ST. PETERSBURG John's Pass Bridge 1957 RPPC",
                "shortTitle": "Antique Florida Postcard ST. PETERSBURG John's Pass 1957 RPPC",
                "Region": "Florida",
                "Country": "USA",
                "City": "St. Petersburg",
                "Era": "Divided Back (1907-1915)",
                "Description": "Depicts fishermen at John's Pass Bridge, a popular tourist and fishing spot. Postcard 
                mentions a family vacation, with references to warm weather and abundant fishing."
            }
        
            Another Example:
            {
                "Title": "Vintage Virginia Postcard NEWPORT NEWS Mariner's Museum 1999",
                "shortTitle": "Vintage Virginia Postcard NEWPORT NEWS 1999",
                "Region": "Virginia",
                "Country": "USA",
                "City": "Newport News",
                "Era": "Photochrome (1945-now)",
                "Description": "Features a museum display, likely sent from a visitor to Newport News. Includes mention of 
                shipbuilding history, with a personal note about travel to Milwaukee."
            }
        
            Another Example:
            {
                "Title": "Vintage Tennessee Postcard MEMPHIS Romeo & Juliet in Cotton Field 1938 RPPC",
                "shortTitle": "Vintage Tennessee Postcard MEMPHIS Cotton Field 1938 RPPC",
                "Region": "Tennessee",
                "Country": "USA",
                "City": "Memphis",
                "Era": "Linen (1930-1945)",
                "Description": "Displays a staged romantic scene of two figures in a cotton field. Likely includes commentary on Southern agriculture or nostalgia, with dated cultural imagery."
            }
        
            If any of the information cannot be found on the postcard, please output just '' for that field.
        
            Always try to put the year in if available. 
        
            Never ever shorten a city name, ie never do New York -> NY. 
        
            Always put the city in all caps in the title field, i.e. 'BOSTON' but never put it in all caps in the City Field.  
            Never put the attraction itself in all caps, ONLY the city. 
        
            Never output any commas within the title.
        
            Never output any sort of formatting block, i.e. ```json just output the raw string.
            
            If the card is blank, don't talk about it being blank, focus on Publisher name or other pertinent info, 
            but don't output anything talking about it being blank, just omit that entirely. It's okay if the description is 
            more concise for a blank card. 
            
            Don't reference a specific year unless it's clearly on the card, default to a decade. 
        
            Try to max out the 65 character limit in the title field, keyword stuff if you must. Never repeat the city or any 
            words within the title ever. 
            The short title can be creatively made, using same formatting guidelines, just make sure it is 1-2 words shorter than the actual first title you wrote.
            Make sure to carefully analyze the **text on the back** of the postcard as well, since it may contain valuable information like the city, region, or country.
            """

            try:
                # The client.models.generate_content pattern is used here
                response = gemini_client.models.generate_content(
                    model='gemini-2.5-flash',  # Or 'gemini-2.5-flash' if available to you
                    contents=[prompt, img_front, img_back],
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": PostcardDetails,
                    },
                )

                logging.debug(f"API Response: {response.text}")
                return response.text

            except Exception as e:
                logging.error(f"Gemini API request failed: {e}")
                return DEFAULT_DETAILS_RESPONSE

    except FileNotFoundError as e:
        logging.error(f"Image file not found: {e}")
        return DEFAULT_DETAILS_RESPONSE
    except Exception as e:
        # This will catch errors if the API key is not set or other client issues
        logging.error(f"Failed to initialize client or open images. Error: {e}")
        return DEFAULT_DETAILS_RESPONSE


def _get_postcard_details_helper(api_key, front_image_path, back_image_path):
    front_image_base64 = encode_image(front_image_path)
    back_image_base64 = encode_image(back_image_path)
    if front_image_base64 is None or back_image_base64 is None:
        logging.error(
            f"Could not encode one of the images for API call in _get_postcard_details_helper. Front: {front_image_path}, Back: {back_image_path}")
        return DEFAULT_DETAILS_RESPONSE
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    prompt = """
    You are given two images of a vintage or antique postcard:

    1. The first image is the **front** of the postcard.
    2. The second image is the **back** of the postcard, which contains text and possibly other relevant details.

    I need you to analyze both the front and back images and provide the following information:

    1. **Title**: Create a descriptive title for the postcard based on the front and back. The title should be **65 
    characters or less**. If the front of the card has a REAL PHOTOGRAPH (not an illustration or print) in it, 
    be sure to add 'RPPC' at the end of the title. Getting the year in there is very important as well. IT IS NOT AN RPPC if you see 
    things like: fine tonal gradation, dot structure typical of photomechanical printing, if it's a collotype or a 
    photo-type print, it is not an RPPC. Also, if it has a publisher name on the back, it is almost never a real 
    photo. Finally RPPC often have visible gloss, but not always.  
    2. **Region**: Identify the U.S. state or region mentioned in the postcard.
    3. **Country**: Identify the country mentioned on the postcard.
    4. **City**: Identify the city or major landmark mentioned on the postcard.
    5. **Era**: You must identify the proper era of the card from these choices: Undivided Back (1901-1907), Divided Back (1907-1915), White Border (1915-1930), Linen (1930-1945), Photochrome (1945-now). You can only choose from those.
    6. **Description** Write a short, descriptive, and non-flowery description of the card, preferably including details that aren't necessarily found in the title, containing elements such as (e.g., "written from a mother to a son" "references to farming" "reference to WWI" "a child's handwriting"). You must also definitively state where the card was sent, the recipients name, address, town, and state/country in the description. 


    Please output the result in the following structure, and NOTHING else:

    Example Output:
    {
        "Title": "Vintage Georgia Postcard SAVANNAH Beach Highway 1983 RPPC",
        "shortTitle": "Vintage Georgia Postcard SAVANNAH Beach 1983 RPPC",
        "Region": "Georgia",
        "Country": "USA",
        "City": "Savannah",
        "Era": "Photochrome (1945-now)",
        "Description": "Features a scenic highway lined with palms and oleanders, likely promoting beach tourism. 
        Sent postmarked from Savannah, with a brief note about a family road trip."
    }

    Another Example:
    {
        "Title": "Antique Wyoming Postcard YELLOWSTONE National Park Gibbon Falls 1913",
        "shortTitle": "Antique Wyoming Postcard YELLOWSTONE Gibbon Falls 1913",
        "Region": "Wyoming",
        "Country": "USA",
        "City": "Yellowstone",
        "Description": "Shows Gibbon Falls, part of a Haynes collection of early park photography. Likely from a 
        traveler describing natural wonders, with references to early park infrastructure."
    }

    Another Example:
    {
        "Title": "Antique Florida Postcard ST. PETERSBURG John's Pass Bridge 1957 RPPC",
        "shortTitle": "Antique Florida Postcard ST. PETERSBURG John's Pass 1957 RPPC",
        "Region": "Florida",
        "Country": "USA",
        "City": "St. Petersburg",
        "Era": "Divided Back (1907-1915)",
        "Description": "Depicts fishermen at John's Pass Bridge, a popular tourist and fishing spot. Postcard 
        mentions a family vacation, with references to warm weather and abundant fishing."
    }

    Another Example:
    {
        "Title": "Vintage Virginia Postcard NEWPORT NEWS Mariner's Museum 1999",
        "shortTitle": "Vintage Virginia Postcard NEWPORT NEWS 1999",
        "Region": "Virginia",
        "Country": "USA",
        "City": "Newport News",
        "Era": "Photochrome (1945-now)",
        "Description": "Features a museum display, likely sent from a visitor to Newport News. Includes mention of 
        shipbuilding history, with a personal note about travel to Milwaukee."
    }

    Another Example:
    {
        "Title": "Vintage Tennessee Postcard MEMPHIS Romeo & Juliet in Cotton Field 1938 RPPC",
        "shortTitle": "Vintage Tennessee Postcard MEMPHIS Cotton Field 1938 RPPC",
        "Region": "Tennessee",
        "Country": "USA",
        "City": "Memphis",
        "Era": "Linen (1930-1945)",
        "Description": "Displays a staged romantic scene of two figures in a cotton field. Likely includes commentary on Southern agriculture or nostalgia, with dated cultural imagery."
    }

    If any of the information cannot be found on the postcard, please output just '' for that field.

    Always try to put the year in if available. 

    Never ever shorten a city name, ie never do New York -> NY. 

    Always put the city in all caps in the title field, i.e. 'BOSTON' but never put it in all caps in the City Field.  
    Never put the attraction itself in all caps, ONLY the city. 

    Never output any commas within the title.

    Never output any sort of formatting block, i.e. ```json just output the raw string.

    Try to max out the 65 character limit in the title field, keyword stuff if you must. Never repeat the city or any 
    words within the title ever. 
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
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()
        details = response_json['choices'][0]['message'][
            'content'] if 'choices' in response_json else "Details not available"
        logging.debug(f"API Response: {details}")
        return details
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return DEFAULT_DETAILS_RESPONSE
    except Exception as e:
        logging.error(f"Exception in _get_postcard_details_helper, skipping call. Error: {e}")
        return DEFAULT_DETAILS_RESPONSE


def get_secondary_postcard_details(api_key, front_image_path, back_image_path, timeout=20, max_workers=10,
                                   max_retries=1, retry_delay=5):
    logging.debug(f"Starting get_secondary_postcard_details for front image: {os.path.basename(front_image_path)}")
    if not api_key:
        logging.error("OpenAI API key is not provided for secondary details API call.")
        return DEFAULT_SECONDARY_RESPONSE

    def api_call():
        return _get_secondary_postcard_details_gemini(front_image_path, back_image_path)

    for attempt in range(max_retries):
        logging.info(f"Attempt {attempt + 1}/{max_retries} for secondary details: {os.path.basename(front_image_path)}")
        result = DEFAULT_SECONDARY_RESPONSE
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future = executor.submit(api_call)
            try:
                result = future.result(timeout=timeout)
                if result != DEFAULT_SECONDARY_RESPONSE:
                    logging.debug(
                        f"Success on attempt {attempt + 1}. get_secondary_postcard_details completed for: {os.path.basename(front_image_path)}")
                    return result
                logging.warning(
                    f"Attempt {attempt + 1} failed (helper returned default response) for secondary details: {os.path.basename(front_image_path)}")
            except concurrent.futures.TimeoutError:
                logging.warning(
                    f"Timeout occurred on attempt {attempt + 1} in get_secondary_postcard_details for: {os.path.basename(front_image_path)}")
            except Exception as e:
                logging.error(
                    f"Exception on attempt {attempt + 1} in get_secondary_postcard_details for: {os.path.basename(front_image_path)}. Error: {e}")
        if attempt < max_retries - 1:
            logging.info(f"Waiting {retry_delay} seconds before next attempt...")
            time.sleep(retry_delay)
        else:
            logging.error(
                f"All {max_retries} attempts failed for secondary details: {os.path.basename(front_image_path)}. Returning default response.")
    return DEFAULT_SECONDARY_RESPONSE


def _get_secondary_postcard_details_gemini(front_image_path, back_image_path):
    try:
        with Image.open(front_image_path) as img_front, Image.open(back_image_path) as img_back:

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
            try:
                # The client.models.generate_content pattern is used here
                response = gemini_client.models.generate_content(
                    model='gemini-2.5-flash',  # Or 'gemini-2.5-flash' if available to you
                    contents=[prompt, img_front, img_back],
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": SecondaryPostcardDetails,
                    },
                )

                logging.debug(f"API Response: {response.text}")
                return response.text

            except Exception as e:
                logging.error(f"Gemini API request failed: {e}")
                return DEFAULT_DETAILS_RESPONSE

    except FileNotFoundError as e:
        logging.error(f"Image file not found: {e}")
        return DEFAULT_DETAILS_RESPONSE
    except Exception as e:
        # This will catch errors if the API key is not set or other client issues
        logging.error(f"Failed to initialize client or open images. Error: {e}")
        return DEFAULT_DETAILS_RESPONSE


def _get_secondary_postcard_details_helper(api_key, front_image_path, back_image_path):
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
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
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



def download_images_from_links(links, tmp_dir):
    logging.info("Starting download_images_from_links")
    postcards = []
    total_links = len(links)
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
            back_link = None
            back_image_filename = None
            back_image_path = None
            if idx + 1 < len(links):
                back_link = links[idx + 1]
                logging.debug(f"Downloading back image from: {back_link}")
                back_response = requests.get(back_link)
                back_response.raise_for_status()
                back_image_filename = f"image_{idx + 1:03d}.jpg"
                back_image_path = os.path.join(tmp_dir, back_image_filename)
                with open(back_image_path, "wb") as f:
                    f.write(back_response.content)
            with postcards_lock:
                postcards.append({
                    "original_index": idx // 2,
                    "front_image_filename": front_image_filename,
                    "back_image_filename": back_image_filename,
                    "front_image_path": front_image_path,
                    "back_image_path": back_image_path,
                    "front_image_link": front_link,
                    "back_image_link": back_link
                })
            logging.debug(f"Finished downloading images for index {idx}")
        except Exception as e:
            logging.error(f"Failed to download images at index {idx}: {e}")
            with postcards_lock:
                postcards.append({
                    "original_index": idx // 2,
                    "front_image_filename": "",
                    "back_image_filename": "",
                    "front_image_path": "",
                    "back_image_path": "",
                    "front_image_link": front_link,
                    "back_image_link": None
                })
        finally:
            with threading.Lock():
                downloaded_count += 2
                progress_percentage = downloaded_count / total_links
                logging.info(f"Download progress: {int(progress_percentage * 100)}%")

    max_workers = min(20, (len(links) + 1) // 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_pair, idx) for idx in range(0, total_links, 2)]
        for future in as_completed(futures):
            pass
    logging.info("Finished download_images_from_links")
    return sorted(postcards, key=lambda x: x["original_index"])


DEFAULT_DETAILS_RESPONSE = '{"Title": "", "Region": "", "Country": "", "City": "", "Era": "", "Description": ""}'
DEFAULT_SECONDARY_RESPONSE = '{"Destination City": "", "Origin City": ""}'


def get_postcard_details(api_key, front_image_path, back_image_path, timeout=20, max_workers=10, max_retries=1,
                         retry_delay=5):
    logging.debug(f"Starting get_postcard_details for front image: {os.path.basename(front_image_path)}")
    if not api_key:
        logging.error("OpenAI API key is not provided for primary details API call.")
        return DEFAULT_DETAILS_RESPONSE

    def api_call():
        return _get_postcard_details_gemini(front_image_path, back_image_path)

    for attempt in range(max_retries):
        logging.info(f"Attempt {attempt + 1}/{max_retries} for primary details: {os.path.basename(front_image_path)}")
        result = DEFAULT_DETAILS_RESPONSE
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future = executor.submit(api_call)
            try:
                result = future.result(timeout=timeout)
                if result != DEFAULT_DETAILS_RESPONSE:
                    logging.debug(
                        f"Success on attempt {attempt + 1}. get_postcard_details completed for: {os.path.basename(front_image_path)}")
                    return result
                logging.warning(
                    f"Attempt {attempt + 1} failed (helper returned default response) for primary details: {os.path.basename(front_image_path)}")
            except concurrent.futures.TimeoutError:
                logging.warning(
                    f"Timeout occurred on attempt {attempt + 1} in get_postcard_details for: {os.path.basename(front_image_path)}")
            except Exception as e:
                logging.error(
                    f"Exception on attempt {attempt + 1} in get_postcard_details for: {os.path.basename(front_image_path)}. Error: {e}")
        if attempt < max_retries - 1:
            logging.info(f"Waiting {retry_delay} seconds before next attempt...")
            time.sleep(retry_delay)
        else:
            logging.error(
                f"All {max_retries} attempts failed for primary details: {os.path.basename(front_image_path)}. Returning default response.")
    return DEFAULT_DETAILS_RESPONSE



def process_batch(api_key, batch):
    logging.info("Starting process_batch")
    postcards_details = []
    for postcard in batch:
        front_image_path = postcard["front_image_path"]
        back_image_path = postcard["back_image_path"]
        front_image_link = postcard["front_image_link"]
        back_image_link = postcard["back_image_link"]
        original_index = postcard["original_index"]
        logging.debug(f"Processing postcard from original index {original_index} in process_details")

        postcard_details = get_postcard_details(api_key, front_image_path, back_image_path)
        secondary_postcard_details = get_secondary_postcard_details(api_key, front_image_path, back_image_path)

        try:
            postcard_details = json.loads(postcard_details)
            secondary_postcard_details = json.loads(secondary_postcard_details)
        except json.JSONDecodeError as e:
            logging.error(
                f"JSONDecodeError in process_batch for postcard at {original_index}: {e}. Details: {postcard_details}, {secondary_postcard_details}")
            postcard_details = {"Title": "", "Region": "", "Country": "", "City": "", "Era": "", "Description": ""}
            secondary_postcard_details = {"Destination City": "", "Origin City": ""}

        postcard_details.update(secondary_postcard_details)

        postcard_details = json.dumps(postcard_details)

        postcards_details.append({
            "original_index": original_index,
            "front_image": postcard["front_image_filename"],
            "back_image": postcard["back_image_filename"],
            "front_image_link": front_image_link,
            "back_image_link": back_image_link,
            "details": postcard_details
        })
    logging.info("Finished process_batch")
    return postcards_details


def get_image_index(filename):
    m = re.search(r'image_(\d+).jpg', filename)
    return int(m.group(1)) if m else -1


def process_postcards_in_folder(api_key, postcards, workers=8):
    logging.info("Starting process_postcards_in_folder")
    total_postcards = len(postcards)
    if total_postcards == 0:
        logging.error("No postcards to process.")
        raise ValueError("No postcards to process.")

    batch_size = max(1, min(total_postcards, math.ceil(total_postcards / workers)))
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
                completed_batches += 1
                progress_value = completed_batches / total_batches
                logging.info(f"Processing batches: {int(progress_value * 100)}%")

    if failed_batches:
        logging.warning(f"Retrying failed batches: {failed_batches}")
        original_failed_batches = list(failed_batches)
        failed_batches = []

        for batch_idx in original_failed_batches:
            batch = postcard_batches[batch_idx]
            try:
                result = process_batch(api_key, batch)
                postcards_details.extend(result)
                completed_batches += 1
                progress_value = completed_batches / total_batches
                logging.info(f"Processing batches (retry): {int(progress_value * 100)}%")
            except Exception as exc:
                logging.error(f"Batch {batch_idx} failed again after retry: {exc}")
                failed_batches.append(batch_idx)

    logging.info("Processing complete!")
    postcards_details = sorted(postcards_details, key=lambda x: x["original_index"])
    logging.info("Finished process_postcards_in_folder")

    return postcards_details


# --- MongoDB connection and upload function ---

def get_mongo_client():
    """Establishes and returns a MongoDB client connection."""
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ismaster')
        logging.info("Successfully connected to MongoDB.")
        return client
    except ConnectionFailure as e:
        logging.error(f"MongoDB connection failed: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while connecting to MongoDB: {e}")
        raise


def upload_to_mongodb(job_id: str, original_links: List[str], csv_data_bytes: bytes):
    """
    Uploads the processed CSV data to MongoDB.
    """
    try:
        client = get_mongo_client()
        db = client[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION_NAME]

        # Use update_one to update the existing job document created in the endpoint
        collection.update_one(
            {"job_id": job_id},
            {"$set": {
                "csv_data": base64.b64encode(csv_data_bytes).decode('utf-8'),
                "timestamp": datetime.utcnow(),  # Store in UTC
                "status": "completed",
                # "filename" and "original_links" are already set at the initial insert
            }},
            upsert=True  # If the job_id doesn't exist for some reason, create it.
        )
        logging.info(f"Job {job_id} CSV data successfully uploaded to MongoDB.")
        client.close()
    except Exception as e:
        logging.error(f"Failed to upload job {job_id} CSV data to MongoDB: {e}")
        raise


# --- FastAPI Application Setup ---

app = FastAPI(
    title="Postcard Processing Backend",
    description="API for processing postcard image links and generating eBay-ready CSVs.",
    version="1.0.0"
)


# Pydantic models for request/response
class ProcessJobRequest(BaseModel):
    links: List[str]


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    timestamp: datetime  # Will be converted to Eastern Time for response
    filename: str
    original_links: Optional[List[str]] = None  # Made optional for listing endpoint
    error_message: Optional[str] = None


# Main processing function to be called as a background task
async def process_job_and_upload(job_id: str, links: List[str]):
    logging.info(f"Starting processing job {job_id} for {len(links)} links.")
    all_rows = []
    try:
        client = get_mongo_client()
        db = client[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION_NAME]

        # FIX: Change to update_one with $set for modifying existing document
        collection.update_one(
            {"job_id": job_id},
            {"$set": {"status": "processing", "timestamp": datetime.utcnow()}},  # Update timestamp to processing start
            upsert=True  # Though it should already exist from process_postcards_endpoint
        )
        client.close()

        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_size = 50

            for i in range(0, len(links), batch_size):
                logging.info(
                    f"Processing batch {i // batch_size + 1} of {math.ceil(len(links) / batch_size)} for job {job_id}")
                current_links_batch = links[i:i + batch_size]

                logging.info(f"Downloading images for batch: {i // batch_size + 1} for job {job_id}")
                postcards = download_images_from_links(current_links_batch, tmp_dir)

                logging.info(f"Processing images for batch: {i // batch_size + 1} for job {job_id}")
                postcards_details = process_postcards_in_folder(OPENAI_API_KEY, postcards, workers=8)

                logging.info(f"Saving CSV data for batch: {i // batch_size + 1} for job {job_id}")
                save_postcards_to_csv(postcards_details, all_rows)

        if not all_rows:
            logging.warning(f"No data generated for job {job_id}. Skipping CSV generation and MongoDB upload.")
            client = get_mongo_client()
            db = client[MONGO_DB_NAME]
            collection = db[MONGO_COLLECTION_NAME]
            collection.update_one(
                {"job_id": job_id},
                {"$set": {"status": "completed_no_data", "timestamp": datetime.utcnow()}},
            )
            client.close()
            return

        df = pd.DataFrame(all_rows)

        df_cleaned = df.copy()
        columns_to_clean = [col for i, col in enumerate(df_cleaned.columns[3:9]) if i + 3 != 7]
        df_cleaned.loc[:, columns_to_clean] = df_cleaned.loc[:, columns_to_clean].applymap(clean_text)
        df_cleaned = df_cleaned.fillna('')

        ebay_ready_df = reformat_for_ebay(df_cleaned)

        csv_buffer = io.StringIO()
        ebay_ready_df.to_csv(csv_buffer, index=False)
        csv_data_bytes = csv_buffer.getvalue().encode('utf-8')

        upload_to_mongodb(job_id, links, csv_data_bytes)
        logging.info(f"Job {job_id} successfully completed and uploaded.")

    except Exception as e:
        logging.error(f"Error processing job {job_id}: {e}", exc_info=True)
        try:
            client = get_mongo_client()
            db = client[MONGO_DB_NAME]
            collection = db[MONGO_COLLECTION_NAME]
            collection.update_one(
                {"job_id": job_id},
                {"$set": {"status": "failed", "error_message": str(e), "timestamp": datetime.utcnow()}},
                upsert=True
            )
            client.close()
        except Exception as db_e:
            logging.error(f"Failed to update job {job_id} status to 'failed' in DB: {db_e}")


import uuid

# Define the Eastern Timezone for conversion
EASTERN_TIMEZONE = pytz.timezone('America/New_York')


# API endpoint to submit a processing job
@app.post("/process-postcards", response_model=JobStatusResponse)
async def process_postcards_endpoint(request: ProcessJobRequest, background_tasks: BackgroundTasks):
    if not request.links:
        raise HTTPException(status_code=400, detail="No links provided.")

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key is not configured on the server.")

    job_id = str(uuid.uuid4())
    logging.info(f"Received request for new job: {job_id}")

    current_utc_time = datetime.utcnow()

    try:
        client = get_mongo_client()
        db = client[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION_NAME]
        initial_status_doc = {
            "job_id": job_id,
            "original_links": request.links,
            "timestamp": current_utc_time,  # Stored in UTC
            "status": "pending",
            "filename": f"postcards_job_{job_id}.csv",
            "csv_data": ""
        }
        collection.insert_one(initial_status_doc)
        client.close()
    except Exception as e:
        logging.error(f"Failed to create initial job record in DB for {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize job. Database error.")

    background_tasks.add_task(process_job_and_upload, job_id, request.links)

    # Convert timestamp to Eastern Time for the response model
    eastern_time = current_utc_time.replace(tzinfo=pytz.utc).astimezone(EASTERN_TIMEZONE)

    return JobStatusResponse(
        job_id=job_id,
        status="pending",
        timestamp=eastern_time,  # Send Eastern Time in response
        filename=f"postcards_job_{job_id}.csv",
        original_links=request.links
    )


# New endpoint to get job status
@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    try:
        client = get_mongo_client()
        db = client[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION_NAME]
        job_document = collection.find_one({"job_id": job_id}, {"_id": 0, "csv_data": 0})
        client.close()

        if job_document:
            # Convert UTC timestamp from DB to Eastern Time for the response
            if 'timestamp' in job_document and isinstance(job_document['timestamp'], datetime):
                utc_time = job_document['timestamp'].replace(tzinfo=pytz.utc)
                job_document['timestamp'] = utc_time.astimezone(EASTERN_TIMEZONE)

            return JobStatusResponse(**job_document)
        else:
            raise HTTPException(status_code=404, detail="Job not found.")
    except Exception as e:
        logging.error(f"Error retrieving job status for {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


# New endpoint to list all jobs
@app.get("/jobs", response_model=List[JobStatusResponse])
async def list_jobs():
    try:
        client = get_mongo_client()
        db = client[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION_NAME]
        # Exclude 'csv_data' and 'original_links' for listing all jobs to keep responses light
        jobs_cursor = collection.find({}, {"_id": 0, "csv_data": 0, "original_links": 0}).sort("timestamp", -1)
        jobs_list = []
        for doc in jobs_cursor:
            # Convert UTC timestamp from DB to Eastern Time for the response
            if 'timestamp' in doc and isinstance(doc['timestamp'], datetime):
                utc_time = doc['timestamp'].replace(tzinfo=pytz.utc)
                doc['timestamp'] = utc_time.astimezone(EASTERN_TIMEZONE)
            jobs_list.append(JobStatusResponse(**doc))
        client.close()
        return jobs_list
    except Exception as e:
        logging.error(f"Error listing all jobs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


# New endpoint to download a specific job's CSV file
@app.get("/jobs/{job_id}/download")
async def download_job_csv(job_id: str):
    try:
        client = get_mongo_client()
        db = client[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION_NAME]
        job_document = collection.find_one({"job_id": job_id})
        client.close()

        if not job_document:
            raise HTTPException(status_code=404, detail="Job not found.")

        if job_document.get("status") != "completed" or not job_document.get("csv_data"):
            raise HTTPException(status_code=400, detail="File not ready for download or processing failed.")

        csv_data_base64 = job_document["csv_data"]
        csv_bytes = base64.b64decode(csv_data_base64)
        filename = job_document.get("filename", f"postcards_job_{job_id}.csv")

        return StreamingResponse(
            io.BytesIO(csv_bytes),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error downloading CSV for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during download.")


if __name__ == "__main__":
    import uvicorn

    logging.info("Starting FastAPI backend...")
    uvicorn.run(app, host="0.0.0.0", port=8000)