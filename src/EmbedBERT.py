import os
import numpy as np
import pandas as pd
from transformers import CLIPTokenizer, CLIPModel
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool
from tqdm import tqdm
from datetime import datetime
from preprocessing import clean_and_label_data, load_and_combine_csv_files
import torch



# Function to extract embeddings using CLIP
def extract_clip_embeddings(model, tokenizer, texts, max_length=77, device='cpu'):
    model.to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for text in tqdm(texts, desc="Generating CLIP embeddings"):
            encoding = tokenizer(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(device)
            output = model.get_text_features(**encoding)  # Extract text embeddings
            embeddings.append(output.cpu().numpy())
    return np.vstack(embeddings)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    transformer_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')

    # Load and prepare data
    directory = "/sise/eliorsu-group/yuvalgor/courses/Data-mining-in-Big-Data/datasets/"
    base_filename = "goodwill_items_job_"
    num_files = 30
    combined_df = load_and_combine_csv_files(directory, base_filename, num_files)
    cleaned_df, _, _ = clean_and_label_data(combined_df)

    # Use log transformation for the target variable
    cleaned_df['currentPrice'] = np.log1p(cleaned_df['currentPrice'])

    # Split data into train, validation, and test
    train_df = cleaned_df.iloc[:100000]
    val_df = cleaned_df.iloc[100000:150000]
    test_df = cleaned_df.iloc[150000:]

    # Extract embeddings for title and description
    train_title_embeddings = extract_clip_embeddings(
        transformer_model, tokenizer, train_df['title'], device=device
    )
    train_desc_embeddings = extract_clip_embeddings(
        transformer_model, tokenizer, train_df['description'], device=device
    )
    val_title_embeddings = extract_clip_embeddings(
        transformer_model, tokenizer, val_df['title'], device=device
    )
    val_desc_embeddings = extract_clip_embeddings(
        transformer_model, tokenizer, val_df['description'], device=device
    )

    # Combine embeddings for title and description
    X_train = np.hstack([train_title_embeddings, train_desc_embeddings])
    y_train = train_df['currentPrice'].values
    X_val = np.hstack([val_title_embeddings, val_desc_embeddings])
    y_val = val_df['currentPrice'].values

    # Train a CatBoost regressor with validation
    train_pool = Pool(data=X_train, label=y_train)
    val_pool = Pool(data=X_val, label=y_val)

    model = CatBoostRegressor(
        iterations=10000,
        learning_rate=0.1,
        depth=6,
        loss_function='RMSE',
        verbose=100
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    test_title_embeddings = extract_clip_embeddings(
        transformer_model, tokenizer, test_df['title'], device=device
    )
    test_desc_embeddings = extract_clip_embeddings(
        transformer_model, tokenizer, test_df['description'], device=device
    )
    X_test = np.hstack([test_title_embeddings, test_desc_embeddings])
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

    model_name = 'CatBoostPriceTextEmbedPrediction'
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
