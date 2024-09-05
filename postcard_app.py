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

# Function to clean and process the title based on the city name
def clean_title(title, city):
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
    city_pattern = re.compile(r'\b' + re.escape(city) + r'\b', re.IGNORECASE)
    if city_pattern.search(title):
        title = city_pattern.sub(city.upper(), title)
    return title

# Function to save postcards details to a CSV file
def save_postcards_to_csv(postcards_details):
    headers = ["front_image", "back_image", "Title", "Region", "Country", "City"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        with open(tmp_file.name, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            for postcard in postcards_details:
                details = eval(postcard["details"])
                title = details.get("Title")
                city = details.get("City", "")
                cleaned_title = clean_title(title, city)
                writer.writerow({
                    "front_image": postcard["front_image"],
                    "back_image": postcard["back_image"],
                    "Title": cleaned_title,
                    "Region": details.get("Region"),
                    "Country": details.get("Country"),
                    "City": details.get("City")
                })
        return tmp_file.name

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

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

    1. **Title**: Create a descriptive title for the postcard based on the front and back. The title should be **80 characters or less**.
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
        "Title": "Vintage Virginia Postcard NEWPORT NEWS Mariner's Museum Cover to Milwaukee 1999",
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

    Never output any sort of formatting block, i.e. ```json just output the raw string.

    Never output the title of the post card directly, i.e. '"Greetings from Weslaco Texas Palm Trees Vintage Postcard"'

    Never output any commas within the title.
    
    Try to max out the 80 character limit in the title field. 

    Make sure to carefully analyze the **text on the back** of the postcard as well, since it may contain valuable information like the city, region, or country.
    """

    payload = {
        "model": "gpt-4o",
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
        "max_tokens": 300
        # "type": "json_object"
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_json = response.json()

    details = response_json['choices'][0]['message']['content'] if 'choices' in response_json else "Details not available"
    print(details)
    return details

# Function for a worker to process its batch of images
def process_batch(api_key, folder_path, image_files):
    postcards_details = []
    for i in range(0, len(image_files), 2):
        if i + 1 < len(image_files):
            front_image_path = os.path.join(folder_path, image_files[i])
            back_image_path = os.path.join(folder_path, image_files[i + 1])
            postcard_details = get_postcard_details(api_key, front_image_path, back_image_path)
            postcards_details.append({
                "front_image": image_files[i],
                "back_image": image_files[i + 1],
                "details": postcard_details
            })
    return postcards_details

# Main function to distribute tasks to workers in parallel
def process_postcards_in_folder(api_key, folder_path, workers=16):
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith((".jpg", ".jpeg", ".png"))])
    total_files = len(image_files)
    batch_size = math.ceil(total_files / workers)
    batches = [image_files[i * batch_size:(i + 1) * batch_size] for i in range(workers - 1)]
    batches.append(image_files[(workers - 1) * batch_size:])
    postcards_details = []
    failed_batches = []  # To store failed batches for reprocessing
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_batch, api_key, folder_path, batch): i for i, batch in enumerate(batches)}
        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                result = future.result()
                postcards_details.extend(result)
            except Exception as exc:
                print(f"Worker {worker_id + 1} generated an exception: {exc}")
                failed_batches.append(worker_id)  # Keep track of failed batches

    # Re-run the failed batches
    if failed_batches:
        print(f"Reprocessing failed batches: {failed_batches}")
        for worker_id in failed_batches:
            batch = batches[worker_id]
            try:
                result = process_batch(api_key, folder_path, batch)
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
    st.write("Upload a ZIP file containing pairs of postcard images (front and back) for processing. Please ensure that they are in sequential order, starting with a card front.")


    api_key = os.getenv("OPENAI_API_KEY")
    zip_file = st.file_uploader("Upload ZIP file", type="zip")

    if api_key and zip_file:
        if "csv_data" not in st.session_state:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Extract zip file
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(tmp_dir)

                with st.spinner("Processing images..."):
                    # Process images using 16 workers
                    postcards_details = process_postcards_in_folder(api_key, tmp_dir, workers=16)

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

