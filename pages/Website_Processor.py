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
from openai import OpenAI


def clean_title(title, city):
    # nothing
    return title


def save_postcards_to_csv(postcards_details):
    headers = ["front_image_link", "back_image_link", "Title", "Date", "Region", "State", "Country", "City",
               "Destination City", "Destination Address", "Recipient", "Year", "Description", "SKU"]
    rows = []
    counter = 1

    for postcard in postcards_details:
        try:
            details = json.loads(postcard["details"])
        except json.JSONDecodeError:
            details = {}  # Handle JSON decoding errors gracefully

        title = details.get("Title", "")
        city = details.get("City", "")
        cleaned_title = clean_title(title, city)
        # Generate SKU
        SKU = '3A-001_' + '{:02d}'.format(counter)
        counter += 1  # Increment counter

        row = {
            "front_image_link": postcard.get("front_image_link", ""),
            "back_image_link": postcard.get("back_image_link", ""),
            "Title": cleaned_title,
            "Date": details.get("Date", ""),
            "State": details.get("State", ""),
            "Country": details.get("Country", ""),
            "City": details.get("City", ""),
            "Destination City": details.get("Destination City", ""),
            "Destination Address": details.get("Destination Address",""),
            "Recipient": details.get("Recipient", ""),
            "Year": details.get("Year", ""),
            "Description": details.get("Description", ""),
            "SKU": SKU
        }
        rows.append(row)

    # No need to sort since they are already in order
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        with open(tmp_file.name, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    return tmp_file.name


def map_and_reorder_csv(input_csv, output_csv):
    # Define original and new headers
    original_headers = ["front_image_link", "back_image_link", "Title", "Date", "Region", "State", "Country", "City",
                        "Destination City", "Destination Address", "Recipient", "Year", "Description", "keywords", "SKU"]

    new_headers = ["Title", "xTags", "Orig. City", "Orig. State", "Orig. Country", "Recipient", "Dest. Street",
                   "Dest. City", "Date", "Year", "Notes", "SKU"]

    # Map original headers to new ones
    header_mapping = {
        "Title": "Title",
        "Description": "Notes",
        "keywords": "xTags",
        "City": "Orig. City",
        "State": "Orig. State",
        "Country": "Orig. Country",
        "Destination Address": "Dest. Street",
        "Destination City": "Dest. City",
        "Recipient": "Recipient",
        "Date": "Date",
        "Year": "Year",
        "SKU": "SKU"
    }

    # Open the input and output CSV files
    with open(input_csv, mode='r', newline='', encoding='utf-8') as infile, \
            open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=new_headers)

        # Write new headers to the output file
        writer.writeheader()

        for row in reader:
            # Create a new row with the new headers
            new_row = {header: '' for header in new_headers}  # Initialize all values as blank

            # Fill in values from the original row based on the mapping
            for orig_header, new_header in header_mapping.items():
                value = row.get(orig_header, '')
                new_row[new_header] = value if value not in [None, 'nan', 'NaN'] else ''  # Replace any 'nan' or None with blank

            # Write the new row to the output file
            writer.writerow(new_row)

            # Write 4 blank rows, but keep the SKU value the same
            for _ in range(4):
                blank_row = {header: '' for header in new_headers}  # Create a blank row
                blank_row["SKU"] = new_row["SKU"]  # Retain the SKU in the blank rows
                writer.writerow(blank_row)



# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def download_images_from_links(links, tmp_dir):
    postcards = []
    # Ensure we process links in pairs
    for idx in range(0, len(links), 2):
        try:
            # Download front image
            front_link = links[idx]
            front_response = requests.get(front_link)
            front_response.raise_for_status()
            front_image_filename = f"image_{idx}.jpg"
            front_image_path = os.path.join(tmp_dir, front_image_filename)
            with open(front_image_path, "wb") as f:
                f.write(front_response.content)

            # Download back image
            back_link = links[idx + 1] if idx + 1 < len(links) else None
            if back_link:
                back_response = requests.get(back_link)
                back_response.raise_for_status()
                back_image_filename = f"image_{idx + 1}.jpg"
                back_image_path = os.path.join(tmp_dir, back_image_filename)
                with open(back_image_path, "wb") as f:
                    f.write(back_response.content)
            else:
                back_image_filename = None
                back_image_path = None

            postcards.append({
                "original_index": idx // 2,
                "front_image_filename": front_image_filename,
                "back_image_filename": back_image_filename,
                "front_image_path": front_image_path,
                "back_image_path": back_image_path,
                "front_link": front_link,
                "back_link": back_link
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
        "Destination City": "Walhalla SC",
        "Destination Address": "3450 West Main Street",
        "Recipient": "Gertrude Smith",
        "Year": "1908",
        "Description": "This vintage postcard, dated March 6, 1908, features Science Hall at the University of South Carolina in Columbia, South Carolina. The front displays an artistic rendering of the impressive classical architecture of the building, surrounded by lush greenery. The back of the postcard, addressed to Gertrude Smith in Walhalla, South Carolina, includes a brief handwritten message from the sender, discussing personal affairs and expressing a longing for quiet, alongside a green one-cent stamp and a postmark from Columbia, SC."
    }

    Another Example:
    {
        "Title": "Boston—Minot Ledge Lighthouse - July 29, 1909 - Vintage Postcard",
        "Date": "July 29",
        "City": "Boston"
        "State": "Massachusetts",
        "Country": "USA",
        "Destination City": "Billings MT",
        "Destination Address": "3250 McBride Avenue",
        "Recipient": "Dresden A Smith",
        "Year": "1909",
        "Description": "This vintage postcard, dated July 29, 1909, depicts the Minot Ledge Lighthouse in Boston, Massachusetts. The front features a beautifully rendered image of the lighthouse under moonlight, casting a serene reflection over the surrounding water. The back of the postcard contains a handwritten message addressing Mr. Dresden A. Smith in Billings, Montana. The sender briefly mentions their visit to Boston, noting they only had a few hours and didn't get a chance to shop for gifts."
    }
    
        Another Example:
    {
        "Title": "Columbia South Carolina—Main Street - July 3, 1908 - Vintage Postcard",
        "Date": "July 3"
        "City": "Columbia"
        "State": "South Carolina",
        "Country": "USA",
        "Destination City": "Walhalla SC",
        "Destination Address": "324 North Roth Boulevard",
        "Recipient": "DA Smith",
        "Year": "1908",
        "Description": "This vintage postcard, postmarked July 3, 1908, showcases a bustling view of Main Street looking from the Capitol in Columbia, South Carolina. The image captures a moment in time with early 20th-century architecture lining the street, pedestrians visible on the sidewalks, and a clear view down the busy thoroughfare. Sent to DA Smith in Walhalla, South Carolina, the postcard features a green one-cent stamp and is a charming artifact from the period, providing a glimpse into the everyday life and urban landscape of Columbia at the time."
    }
    
    If any of the information cannot be found on the postcard, please output just "" for that field. You can infer or guess if you feel you have enough information. 

    Always try to put the year in if available. 
    
    You MUST ensure that the destination city field has the city and state abbreviation, never the whole state name written out, i.e. Liberty NY, Great Falls MT, etc.  
    
    Never put a year or another month in the day datapoint, always output (Month Day) i.e May 16
    
    Always output the full, un-abbreviated address, i.e. never shorten Street to St. for example. Also, ensure that 
    directions (North, West, etc.) are not abbreviated. 

    Never output any sort of formatting block, i.e. ```json just output the raw string.
    
    Never output bad spacing in the titles, i.e. if you transcribe a card as 'VirginiaUniversity' write it in the title as 'Virginia University' with proper spacing. 
    
    Never discuss any missing information in the description, just write it as if there is nothing missing. 

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


import time  # For adding delays between retries if necessary

# Updated process_batch function with retry logic
def process_batch(api_key, batch, retries=3):
    postcards_details = []
    failed_postcards = []

    for postcard in batch:
        front_image_path = postcard["front_image_path"]
        back_image_path = postcard["back_image_path"]
        front_image_link = postcard["front_link"]
        back_image_link = postcard["back_link"]
        original_index = postcard["original_index"]
        attempts = 0
        success = False

        # Retry logic for individual postcard
        while attempts < retries and not success:
            try:
                postcard_details = get_postcard_details(api_key, front_image_path, back_image_path)
                postcards_details.append({
                    "original_index": original_index,
                    "front_image": postcard["front_image_filename"],
                    "back_image": postcard["back_image_filename"],
                    "front_image_link": front_image_link,
                    "back_image_link": back_image_link,
                    "details": postcard_details,
                })
                success = True  # Successfully processed postcard
            except Exception as exc:
                attempts += 1
                print(f"Postcard at index {original_index} failed attempt {attempts}: {exc}")
                if attempts >= retries:
                    # Track failed postcards after max retries
                    failed_postcards.append({
                        "original_index": original_index,
                        "error": str(exc),
                        "postcard": postcard
                    })
                else:
                    time.sleep(1)  # Optional: Add a delay before retrying

    return postcards_details, failed_postcards

# Updated process_postcards_in_folder function
def process_postcards_in_folder(api_key, postcards, workers=10, retries=3):
    total_postcards = len(postcards)
    if total_postcards == 0:
        raise ValueError("No postcards to process.")

    # Split postcards into batches
    batch_size = max(1, total_postcards // workers)
    batches = [postcards[i:i + batch_size] for i in range(0, total_postcards, batch_size)]

    postcards_details = []
    failed_postcards = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_batch, api_key, batch, retries): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            try:
                batch_valid_postcards, batch_failed_postcards = future.result()
                postcards_details.extend(batch_valid_postcards)
                failed_postcards.extend(batch_failed_postcards)
            except Exception as exc:
                print(f"A batch generated an exception: {exc}")

    # Sort postcards by their original index to maintain order
    postcards_details = sorted(postcards_details, key=lambda x: x["original_index"])

    # Optionally log or handle failed postcards
    if failed_postcards:
        print(f"Number of postcards that failed after {retries} retries: {len(failed_postcards)}")
        for failed in failed_postcards:
            print(f"Postcard at index {failed['original_index']} failed: {failed['error']}")

    return postcards_details, failed_postcards


# Function to analyze a row using the AI model
def analyze_row_with_ai(row, headers):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # Convert the row into a string excluding 'keywords' since it doesn't exist yet
    row_string = ", ".join([f"{header}: {row[header]}" for header in headers if header != "keywords" or
                            "front_image_link" or "back_image_link"])
    print(row_string)


    keyword_list = """
    1900s, 1910s, 1920s, 1930s, 1940s, 1950s, 1960s, 1970s, Advertising, Aerial, Africa, Agriculture, Airbrushed, Airplane, Albertype, American South, Americana, Amusement Park, Ancient World, Animal, Antique, Antique, Antiquity, Arch, Archaeology, Architecture, Arctic, Art, Art Deco, Art Nouveau, Artist Signed, Astronomy, Autumn, Aviation, Bamforth, Bank, Bar, Baseball, Bathing, Beach, Bear, Beautiful, Beer, Bike, Bird, Bird's Eye, Birds Eye, Birthday, Black, Black & White, Blue, Boardwalk, Boat, Bob Petley, Bold Color, Border, Bow & Arrow, Brick, Bridge, Brown, Bullfighting, Bus, Cabarret, Cabin, Cactus, Calligraphy, Cameo, Camera, Canal, Canoe, Cape Cod, Capitol, Car, Caribbean, Carriage, Cartoon, Casino, Castle, Cat, Cathedral, Catholic, Cemetery, Cheese, Children, Christmas, Church, Citrus, Classical, Clothing, Clouds, Clover, Coca Cola Sign, Coffee, College, Colonial, Colorful, Column, Comic, Cow, Cowboy, Crab, Cross-section, Curt Teich, Cute, Dam, Dancing, Dark, Delivery Cart, Depot, Desert, Detailed, Disaster, Dog, Dogs, Dome, Domestic, DPO, Dressed Animal, Drinking, Drum, Dutch Kids, Earl Christy, Embossed, Engineering, Error, Ethnographic, European, Exaggeration, Fan, Fantasy, Farm, Fashion, Fat, Field, Fire, Fireplace, Fish, Fishing, Flag, Floral, Flower, Font, Food, Football, Forest, Fountain, Frashers, French Riviera, Front, Frontier, Fruit, Funny, Funny Message, Garden, Gas Station, Geology, Girl, Glacier, Globe, Gold, Golf, Gothic, Graphic, Gray, Great Lakes, Green, Green Pen, Grogan, Hand Colored, Handwriting, Hat, Highway, Hiking, History, Holiday, Horse, Horse Cart, Horse Racing, Hotel, House, Humor, Ice, Iconic, Image Only, Immigrant Message, Immigration, In Stock, Indian, Industrial, Interesting Border, Interesting Cancel, Interesting Message, Interior, Island, Ivy, Jazz, Jersey Shore, Jungle, Kid, Kid Writing, Kraemer, Kropp, Labor, Lake, Large Letter, Law, Leaf, Leighton, Leisure, Library, Light House, Lighthouse, Linen, Lion, Liquor, Lobster, Log Cabin, Logging, Love, Main Street, Man, Map, Maritime, Market, Men, Message, Mid-Atlantic, Midwest, Midwest, Military, Mining, Mississippi River, Money, Monument, Moon, Moonlight, Mountain West, Mountains, Multiview, Museum, Music, Mustache, Name, National Park, Nationality, Native American, Natural Wonder, Nature, Navy, Neon Sign, New, New England, New England, Newman, Night, Nightclub, Non-geographic, Novelty, Nude, NYC, Occult, Occupation, Ocean, Oil, Old West, Orange, Orcean, Orchard, Out Of Stock, Outdoor Activities, Pacific Northwest, Palm Tree, Parade, Paris, Park, Pastels, Patriotic, Peaceful, Pennant, People, Piano, Picnic, Pink, Poem, Political, Pond, Pool, Portrait, Portrait Orientation, Poster Art, Potato, Primate, Prison, Private Mailing Card, Profane, Pun, Purple, Purple Pen, Railroad, Rainbow, Real Photo, Red, Red Pen, Reflections, Religion, Reptile, Restaurant, Risque, River, Roadside, Rock Climbing, Rockies, Rodeo, Romance, Rose, Rotograph, Route 1, Route 66, Ruins, Sailing, Sand, Saying, Sculpture, Seafood, Sepia, Sex, Ship, Silhouette, Silver, Skiing, Sky, Skyline, Skyscraper, Sleep, Smokestack, Smoking, Snow, Social History, Soldier Mail, South, Southwest, Sports, Spring, Stadium, Stars, Station Wagon, Statue, Strange, Subsurface, Subway, Summer, Sun, Sunset, Surfing, Swamp, Swimming, Taxidermy, Tea, Technology History, Telephone, Telescope, Tennis, Text, Textile, Theater, Thumbprint, Tichnor, Tobacco, Toboggan, Toilet, Top Hat, Tower, Train, Train Station, Transportation, Tree, Trolley, Trolley Train, Tropical, Twilight, Umbrella, University, Urban, Vegetable, Vineyard, Vintage, Vintage, Volcano, Water, Water Sports, Waterfalls, Weird, West, West Coast, White, Windmill, Wine, Winter, Winter Sports, Woman, WWII, WWII Soldier Mail, Yellow, Zoo
    """

    prompt = f"""
        You are a keyword picker. Youi will examine a description of a postcard, and pick from the list of keywords 
        given to you, you will choose the top 3-7 most relevant from that list and output them, comma separated like 
        this: 
        Example 1:
        Volcano, Tree, Yellow 

        Example 2: 
        Summer, Southwest, Social History, Embossed
        
        DO NOT UNDER ANY CIRCUMSTANCES OUTPUT ANY KEYWORD THAT ISN't IN THE LIST BELOW. YOU HAVE FAILED YOUR TASK IF 
        YOU DO SO.  

        You will never output anything else. 
        
        Here are the list of keywords you can choose from. You will never output any other keyword. Never put the 
        city name in the keywords.
        
        Here are the ONLY keywords you can choose from:
        {keyword_list} 
        """

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Here is the row you'll be analyzing:\n{row_string}"},
        ]
    )

    keywords = response.choices[0].message.content
    print(keywords)

    return keywords


def process_row_batch(rows, headers):
    processed_rows = []
    for row in rows:
        try:
            # Analyze the row using AI to get keywords
            keywords = analyze_row_with_ai(row, headers)
            # Add the keywords to the row
            row["keywords"] = keywords
        except Exception as e:
            print(f"Error processing row {row.get('original_index', '')}: {e}")
            row["keywords"] = ""  # Optionally, set to empty string or handle accordingly
        processed_rows.append(row)
    return processed_rows



def process_csv_with_keywords(input_csv, output_csv, workers=10):
    # Read the input CSV
    with open(input_csv, mode="r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        headers = reader.fieldnames  # Read existing headers

        if "keywords" not in headers:
            headers.append("keywords")  # Add 'keywords' column only when writing the output

        # Add original index to each row to track its position
        rows = [{**row, "original_index": idx} for idx, row in enumerate(reader)]

    # Split rows into batches for each worker
    batch_size = max(1, len(rows) // workers)
    row_batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]

    # Use ThreadPoolExecutor for parallel processing
    processed_rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_row_batch, batch, headers): idx for idx, batch in enumerate(row_batches)}

        # Collect results as they complete
        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                batch_result = future.result()
                processed_rows.extend(batch_result)
            except Exception as exc:
                print(f"An error occurred during processing batch {batch_idx}: {exc}")

    # Sort processed rows by the original index to maintain order
    processed_rows = sorted(processed_rows, key=lambda x: x["original_index"])

    # Remove the 'original_index' from the rows before writing to CSV
    for row in processed_rows:
        row.pop("original_index", None)

    # Write the updated rows with keywords to a new CSV
    with open(output_csv, mode="w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(processed_rows)

    print(f"Updated CSV with keywords saved to {output_csv}")


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
                    postcards = download_images_from_links(links, tmp_dir)

                with st.spinner("Processing images..."):
                    # Process postcards using workers
                    postcards_details, failed_postcards = process_postcards_in_folder(api_key, postcards, workers=10)

                    # Save the results to a CSV file
                    csv_file = save_postcards_to_csv(postcards_details)

                with st.spinner("Adding keywords..."):
                    process_csv_with_keywords(csv_file, csv_file)
                    st.write("Processing complete!")

                    import unicodedata
                    import re
                    import pandas as pd

                    # Function to clean and normalize text
                    def clean_text(text):
                        # Normalize the text to decompose accents
                        text = unicodedata.normalize('NFKD', str(text))
                        # Remove any non-Latin characters
                        text = ''.join([c for c in text if unicodedata.category(c) != 'Mn'])
                        # Remove unwanted symbols, keeping basic punctuation and alphanumeric characters
                        text = re.sub(r'[^A-Za-z0-9\s.,?!\'"<>-_]', '', text)
                        return text

                    df = pd.read_csv(csv_file)

                    # Apply the cleaning function to all columns
                    df_cleaned = df.applymap(clean_text)

                    # Save the cleaned data to a new CSV file
                    df_cleaned.to_csv('cleaned_file.csv', index=False)

                    map_and_reorder_csv('cleaned_file.csv', 'formatted_file.csv')

                    # Read the CSV file and store data in session state
                    with open('formatted_file.csv', "rb") as f:
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
