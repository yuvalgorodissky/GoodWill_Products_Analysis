import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from scipy.sparse import hstack, coo_matrix, csr_matrix
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import os
from datetime import datetime
from preprocessing import load_and_combine_csv_files, clean_and_label_data

class GoodwillDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_image_from_url(self, url):
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            return self.image_transforms(img)
        except:
            return torch.zeros(3, 224, 224)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        title = str(self.df.iloc[idx]['title'])
        description = str(self.df.iloc[idx]['description'])
        
        image_url = str(self.df.iloc[idx]['imageUrls'])
        image_tensor = self.load_image_from_url(image_url)
        
        title_encoding = self.tokenizer(
            title,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        desc_encoding = self.tokenizer(
            description,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        price = float(self.df.iloc[idx]['currentPrice'])
        
        return {
            'title_input_ids': title_encoding['input_ids'].squeeze(),
            'title_attention_mask': title_encoding['attention_mask'].squeeze(),
            'desc_input_ids': desc_encoding['input_ids'].squeeze(),
            'desc_attention_mask': desc_encoding['attention_mask'].squeeze(),
            'image': image_tensor,
            'price': torch.tensor(price, dtype=torch.float)
        }

class PricePredictor(nn.Module):
    def __init__(self, transformer_model, image_encoder):
        super().__init__()
        
        self.transformer = transformer_model
        self.transformer_dim = self.transformer.config.hidden_size
        
        self.image_encoder = image_encoder
        self.image_dim = 2048  # ResNet50 output dimension
        
        self.text_fusion = nn.Sequential(
            nn.Linear(self.transformer_dim * 2, self.transformer_dim),
            nn.LayerNorm(self.transformer_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(self.transformer_dim + self.image_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        self.final_activation = nn.ReLU()
        
    def forward(self, title_input_ids, title_attention_mask, 
                desc_input_ids, desc_attention_mask, images):
        title_output = self.transformer(
            input_ids=title_input_ids,
            attention_mask=title_attention_mask
        )
        
        desc_output = self.transformer(
            input_ids=desc_input_ids,
            attention_mask=desc_attention_mask
        )
        
        title_features = title_output.last_hidden_state[:, 0, :]
        desc_features = desc_output.last_hidden_state[:, 0, :]
        
        text_combined = torch.cat([title_features, desc_features], dim=1)
        text_fused = self.text_fusion(text_combined)
        
        image_features = self.image_encoder(images)
        image_features = image_features.view(image_features.size(0), -1)
        
        multimodal_features = torch.cat([text_fused, image_features], dim=1)
        fused_features = self.multimodal_fusion(multimodal_features)
        
        output = self.regressor(fused_features)
        return self.final_activation(output).squeeze()

def count_parameters_by_layer(model):
    print("\nParameters by Layer:")
    print("-" * 50)
    
    bert_params = sum(p.numel() for p in model.transformer.parameters())
    print(f"1. BERT Encoder: {bert_params:,} parameters")
    
    resnet_params = sum(p.numel() for p in model.image_encoder.parameters())
    print(f"2. ResNet Encoder: {resnet_params:,} parameters")
    
    print("\n3. Text Fusion Layer:")
    for i, layer in enumerate(model.text_fusion):
        layer_params = sum(p.numel() for p in layer.parameters())
        print(f"   {i+1}. {layer.__class__.__name__}: {layer_params:,} parameters")
    
    print("\n4. Multimodal Fusion Layer:")
    for i, layer in enumerate(model.multimodal_fusion):
        layer_params = sum(p.numel() for p in layer.parameters())
        print(f"   {i+1}. {layer.__class__.__name__}: {layer_params:,} parameters")
    
    print("\n5. Regressor Layers:")
    for i, layer in enumerate(model.regressor):
        layer_params = sum(p.numel() for p in layer.parameters())
        print(f"   {i+1}. {layer.__class__.__name__}: {layer_params:,} parameters")
    
    total_params = sum(p.numel() for p in model.parameters())
    print("\n" + "-" * 50)
    print(f"Total Parameters: {total_params:,}")

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader)
    for batch in progress_bar:
        title_input_ids = batch['title_input_ids'].to(device)
        title_attention_mask = batch['title_attention_mask'].to(device)
        desc_input_ids = batch['desc_input_ids'].to(device)
        desc_attention_mask = batch['desc_attention_mask'].to(device)
        images = batch['image'].to(device)
        price = batch['price'].to(device)
        
        optimizer.zero_grad()
        output = model(title_input_ids, title_attention_mask,
                      desc_input_ids, desc_attention_mask,
                      images)
        
        loss = criterion(output, price)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        avg_loss = total_loss / (progress_bar.n + 1)
        progress_bar.set_description(f'Loss: {avg_loss:.4f}')
    
    return total_loss / len(train_loader)

def evaluate(model, test_loader, device):
    model.eval()
    predictions = []
    actuals = []
    titles = []
    descriptions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            title_input_ids = batch['title_input_ids'].to(device)
            title_attention_mask = batch['title_attention_mask'].to(device)
            desc_input_ids = batch['desc_input_ids'].to(device)
            desc_attention_mask = batch['desc_attention_mask'].to(device)
            images = batch['image'].to(device)
            price = batch['price']
            
            output = model(title_input_ids, title_attention_mask,
                         desc_input_ids, desc_attention_mask,
                         images)
            
            predictions.extend(output.cpu().numpy())
            actuals.extend(price.numpy())
            
            titles.extend(tokenizer.batch_decode(title_input_ids, skip_special_tokens=True))
            descriptions.extend(tokenizer.batch_decode(desc_input_ids, skip_special_tokens=True))
    
    return np.array(predictions), np.array(actuals), titles, descriptions

def main():
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loading parameters
    directory = "/sise/eliorsu-group/yuvalgor/courses/Data-mining-in-Big-Data/datasets/"
    base_filename = "goodwill_items_job_"
    num_files = 30

    # Load and preprocess data
    combined_df = load_and_combine_csv_files(directory, base_filename, num_files)
    cleaned_df, le_state, le_category = clean_and_label_data(combined_df)
    train_val_df, test_df = train_test_split(cleaned_df, test_size=400000, random_state=42)

    # Initialize models
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    transformer_model = AutoModel.from_pretrained('bert-base-uncased')
    
    image_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    image_model = nn.Sequential(*list(image_model.children())[:-1])

    # Create datasets and dataloaders
    train_dataset = GoodwillDataset(train_val_df, tokenizer)
    test_dataset = GoodwillDataset(test_df, tokenizer)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    model = PricePredictor(
        transformer_model=transformer_model,
        image_encoder=image_model
    ).to(device)

    count_parameters_by_layer(model)

    # Setup training
    criterion = nn.MSELoss()
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if 'transformer' not in n and 'image_encoder' not in n],
            'lr': 1e-3
        },
        {
            'params': [p for n, p in model.named_parameters() if 'transformer' in n or 'image_encoder' in n],
            'lr': 1e-5
        }
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=2,
        verbose=True
    )

    # Training loop
    num_epochs = 2
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)

    # Evaluation
    predictions, actuals, titles, descriptions = evaluate(model, test_loader, device)

    # Save model and results
    model_name = 'ImageTextFusion'
    save_dir = '/sise/eliorsu-group/yuvalgor/courses/Data-mining-in-Big-Data/models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)

    print("\nTest Set Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Save model
    model_path = os.path.join(save_dir, f'{model_name}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_mse': mse,
        'test_rmse': rmse,
        'test_r2': r2,
        'tokenizer': tokenizer,
    }, model_path)
    print(f'\nModel saved at: {model_path}')

    # Create analysis DataFrame
    test_df['predicted_price'] = predictions
    test_df['actual_price'] = test_df['currentPrice']
    test_df['price_difference'] = test_df['predicted_price'] - test_df['actual_price']
    test_df['price_difference_pct'] = ((test_df['actual_price'] - test_df['predicted_price']) / test_df['predicted_price']) * -100

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
    analysis_df = analysis_df.sort_values('price_difference', ascending