import requests
import re
from html import unescape
from tqdm import tqdm
import pandas as pd
import argparse
BASE_URL = "https://buyerapi.shopgoodwill.com/api/ItemDetail/GetItemDetailModelByItemId/"
def fetch_item_details(item_id):
    """Fetch item details from the API for a given item ID."""
    url = f"{BASE_URL}{item_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def clean_html(raw_html):
    """Remove HTML tags and unescape HTML entities in the input string."""
    if raw_html is None:
        return ""
    clean_text = re.sub('<.*?>', '', raw_html)
    return unescape(clean_text)


def format_image_urls(image_url_string, image_server):
    """Format and return the first image URL if available."""
    if not image_url_string or not image_server:
        return None  # Return None if no image or server is provided

    # Replace backslashes with forward slashes and split the string
    urls = image_url_string.replace("\\", "/").split(';')

    if urls:  # Check if there are URLs available
        # Construct and return the first full URL
        return f"{image_server}{urls[0]}"

    return None  # Return None if no valid URLs are found


def extract_category_hierarchy(category_breadcrumbs):
    """Extract main and subcategory from the category breadcrumbs."""
    main_category = None
    sub_category = None

    if category_breadcrumbs:
        if len(category_breadcrumbs) > 0:
            main_category = category_breadcrumbs[0]['name']
        if len(category_breadcrumbs) > 1:
            sub_category = category_breadcrumbs[1]['name']

    return main_category, sub_category

def clean_description(description):
    """Clean and truncate the description to remove content after 'NOTES'."""
    if description:
        cleaned = clean_html(description)
        notes_index = cleaned.upper().find("NOTES")
        if notes_index != -1:
            cleaned = cleaned[:notes_index].strip()
        return " ".join(cleaned.split())  # Remove excessive spaces and newlines
    return ""

def process_item(item_id):
    """Process a single item and return its details as a dictionary."""
    item_details = fetch_item_details(item_id)
    if item_details:
        cleaned_description = clean_description(item_details.get("description", ""))
        image_url = format_image_urls(item_details.get("imageUrlString", ""), item_details.get("imageServer", ""))
        main_category, sub_category = extract_category_hierarchy(item_details.get("categoryBreadCrumbs", []))

        return {
            "itemId": item_details.get("itemId"),
            "title": item_details.get("title"),
            "mainCategory": main_category,
            "subCategory": sub_category,
            "categoryId": item_details.get("categoryId"),
            "categoryName": item_details.get("categoryParentList"),
            "currentPrice": item_details.get("currentPrice"),
            "numberOfBids": item_details.get("numberOfBids"),
            "remainingTime": item_details.get("remainingTime"),
            "description": cleaned_description,
            "imageUrls": image_url,
            "pickupCity": item_details.get("pickupCity"),
            "pickupState": item_details.get("pickupState"),
            "sellerCompanyName": item_details.get("sellerCompanyName"),
            "sellerLandingPageName": item_details.get("sellerLandingPageName"),
            "keyWords": item_details.get("keyWords"),
        }
    return None

def collect_items(start_item_id=218000000, max_items=100000, save_interval=10000, output_file="collected_items.csv"):
    """Collect items starting from a given item ID and save in chunks to avoid memory issues."""
    collected_data = []
    header_written = False  # Track whether the header has been written

    for i, item_id in enumerate(tqdm(range(start_item_id, start_item_id + max_items), desc="Collecting items")):
        try:
            item_data = process_item(item_id)
            if item_data:
                collected_data.append(item_data)
        except Exception as e:
            print(f"Error processing item {item_id}: {e}")

        if (i + 1) % save_interval == 0:
            # Save collected data to file and clear the list
            pd.DataFrame(collected_data).to_csv(output_file, mode='a', header=not header_written, index=False)
            collected_data = []
            header_written = True  # Ensure header is not written again

    # Save any remaining data
    if collected_data:
        pd.DataFrame(collected_data).to_csv(output_file, mode='a', header=not header_written, index=False)
def main():
    """Main function to initiate data collection."""
    parser = argparse.ArgumentParser(description="Collect items from Goodwill API and save to CSV.")
    parser.add_argument("--start_item_id",default=218000000, type=int, help="Starting item ID.")
    parser.add_argument("--max_items", type=int, default=500000, help="Number of items to collect.")
    parser.add_argument("--save_interval", type=int, default=10000, help="Number of items to save before writing to file.")
    parser.add_argument("--output_file", type=str, default="/sise/eliorsu-group/yuvalgor/courses/Data-mining-in-Big-Data/datasets/collected_items.csv", help="Output CSV file.")

    args = parser.parse_args()

    print("Starting data collection...")
    collect_items(start_item_id=args.start_item_id, max_items=args.max_items, save_interval=args.save_interval, output_file=args.output_file)
    print("Data collection complete. Data saved in chunks.")

if __name__ == "__main__":
    main()
