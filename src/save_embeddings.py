import os
import numpy as np
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO
import torch
from preprocessing import load_and_combine_csv_files

def extract_clip_embeddings(model, processor, texts_or_images, is_text=True, device='cpu'):
    model.to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for item in tqdm(texts_or_images, desc="Generating CLIP embeddings"):
            if is_text:
                inputs = processor(
                    text=item,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(device)
                outputs = model.get_text_features(**inputs)
            else:
                inputs = processor(
                    images=item,
                    return_tensors="pt"
                ).to(device)
                outputs = model.get_image_features(**inputs)
            embeddings.append(outputs.cpu().numpy().flatten())
    return np.vstack(embeddings)


def extract_image_embeddings_from_urls(model, processor, urls, device='cpu'):
    """
    Load images from URLs and extract their CLIP embeddings in a single pass

    Args:
        model: CLIP model
        processor: CLIP processor
        urls: List of image URLs
        device: Computing device ('cpu' or 'cuda')

    Returns:
        List of image embeddings
    """
    model.to(device)
    model.eval()
    embeddings = []

    with torch.no_grad():
        for url in tqdm(urls, desc="Processing images"):
            try:
                # Load image from URL
                response = requests.get(url)
                img = Image.open(BytesIO(response.content)).convert("RGB")

                # Process image and get embedding
                inputs = processor(
                    images=img,
                    return_tensors="pt"
                ).to(device)
                outputs = model.get_image_features(**inputs)
                embeddings.append(outputs.cpu().numpy().flatten())

            except Exception as e:
                print(f"Error processing URL: {url}")
                print(f"Error message: {str(e)}")
                # Add a zero vector as embedding for failed items
                embeddings.append(np.zeros(512))  # CLIP base model output dimension is 512

    return np.vstack(embeddings).tolist()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize CLIP model and processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    # Data loading parameters
    directory = "/sise/eliorsu-group/yuvalgor/courses/Data-mining-in-Big-Data/datasets/"
    base_filename = "goodwill_items_job_"
    num_files = 30

    # Load and preprocess data
    combined_df = load_and_combine_csv_files(directory, base_filename, num_files)

    # Extract text embeddings for titles and descriptions
    combined_df['title_embedding'] = extract_clip_embeddings(
        clip_model, processor, combined_df['title'], is_text=True, device=device
    ).tolist()

    combined_df['description_embedding'] = extract_clip_embeddings(
        clip_model, processor, combined_df['description'], is_text=True, device=device
    ).tolist()


    # Extract image embeddings directly from URLs
    combined_df['image_embedding'] = extract_image_embeddings_from_urls(
        clip_model, processor, combined_df['imageUrls'], device=device
    )

    # Save the DataFrame with embeddings
    results_dir = '/sise/eliorsu-group/yuvalgor/courses/Data-mining-in-Big-Data/datasets/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    output_file = os.path.join(results_dir, "combined_embeddings.csv")
    combined_df.to_csv(output_file, index=False)
    print(f"DataFrame with embeddings saved at {output_file}")

if __name__ == "__main__":
    main()