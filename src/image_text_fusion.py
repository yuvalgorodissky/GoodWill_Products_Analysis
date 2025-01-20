import os
import numpy as np
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool
from tqdm import tqdm
from datetime import datetime
from preprocessing import load_and_combine_csv_files, clean_and_label_data
from PIL import Image
import requests
from io import BytesIO
import torch


# Function to extract embeddings using CLIP
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
            embeddings.append(outputs.cpu().numpy())
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
    cleaned_df, _, _ = clean_and_label_data(combined_df)

    # Use log transformation for the target variable
    cleaned_df['currentPrice'] = np.log1p(cleaned_df['currentPrice'])

    # Split data into train, validation, and test sets
    train_df = cleaned_df.iloc[:100000]
    val_df = cleaned_df.iloc[100000:150000]
    # Select the last 150K rows
    test_df = cleaned_df.iloc[-150000:]

    # Extract text embeddings for titles and descriptions
    train_image_embeddings = extract_image_embeddings_from_urls(model=clip_model, processor=processor,
                                                                urls=train_df['imageUrls'], device=device)
    val_image_embeddings = extract_image_embeddings_from_urls(model=clip_model, processor=processor,
                                                              urls=val_df['imageUrls'], device=device)
    train_title_embeddings = extract_clip_embeddings(
        clip_model, processor, train_df['title'], is_text=True, device=device
    )
    train_desc_embeddings = extract_clip_embeddings(
        clip_model, processor, train_df['description'], is_text=True, device=device
    )
    val_title_embeddings = extract_clip_embeddings(
        clip_model, processor, val_df['title'], is_text=True, device=device
    )
    val_desc_embeddings = extract_clip_embeddings(
        clip_model, processor, val_df['description'], is_text=True, device=device
    )

    # Extract image embeddings


    # Combine embeddings for multimodal input
    X_train = np.hstack([train_title_embeddings, train_desc_embeddings, train_image_embeddings])
    y_train = train_df['currentPrice'].values
    X_val = np.hstack([val_title_embeddings, val_desc_embeddings, val_image_embeddings])
    y_val = val_df['currentPrice'].values


    # Train a CatBoost regressor
    train_pool = Pool(data=X_train, label=y_train)
    val_pool = Pool(data=X_val, label=y_val)

    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.1,
        depth=6,
        loss_function='RMSE',
        verbose=100
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    # Make predictions
    test_title_embeddings = extract_clip_embeddings(
        clip_model, processor, test_df['title'], is_text=True, device=device
    )
    test_desc_embeddings = extract_clip_embeddings(
        clip_model, processor, test_df['description'], is_text=True, device=device
    )

    test_image_embeddings = extract_image_embeddings_from_urls(
        model=clip_model, processor=processor, urls=test_df['imageUrls'], device=device
    )
    X_test = np.hstack([test_title_embeddings, test_desc_embeddings, test_image_embeddings])
    y_test = test_df['currentPrice'].values
    predictions = model.predict(X_test)

    # Convert predictions and actual values back to the original price scale
    predictions = np.expm1(predictions)
    actuals = np.expm1(y_test)

    # Compute metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)

    print("\nTest Set Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Save model
    save_dir = '/sise/eliorsu-group/yuvalgor/courses/Data-mining-in-Big-Data/models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_name = 'CLIPCatBoostPricePrediction'
    model_path = os.path.join(save_dir, f'{model_name}.cbm')
    model.save_model(model_path)
    print(f'\nModel saved at: {model_path}')

    # Analysis and saving results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    test_df['predicted_price'] = predictions
    test_df['actual_price'] = np.expm1(test_df['currentPrice'])  # Revert log transformation
    test_df['price_difference'] = test_df['predicted_price'] - test_df['actual_price']
    test_df['price_difference_pct'] = ((test_df['actual_price'] - test_df['predicted_price']) / test_df[
        'predicted_price']) * -100

    analysis_df = test_df[[
        'title',
        'actual_price',
        'predicted_price',
        'price_difference',
        'price_difference_pct',
        'mainCategory',
        'description',
        'pickupState',
        'imageUrls',
        'itemId'
    ]].copy()

    numeric_cols = ['actual_price', 'predicted_price', 'price_difference', 'price_difference_pct']
    analysis_df[numeric_cols] = analysis_df[numeric_cols].round(2)
    analysis_df = analysis_df.sort_values('price_difference', ascending=False)

    results_dir = '/sise/eliorsu-group/yuvalgor/courses/Data-mining-in-Big-Data/results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    filename = f"{model_name}_analysis_{timestamp}.csv"
    save_path = os.path.join(results_dir, filename)
    analysis_df.to_csv(save_path, index=False)
    print(f"Analysis results saved at {save_path}")

if __name__ == "__main__":
    main()
