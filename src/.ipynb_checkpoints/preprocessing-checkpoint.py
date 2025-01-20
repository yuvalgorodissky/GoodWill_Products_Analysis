import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
def load_and_combine_csv_files(directory, base_filename, num_files):
    """
    Load and combine multiple CSV files into a single DataFrame.

    Parameters:
        directory (str): The directory containing the CSV files.
        base_filename (str): The base filename for the CSV files.
        num_files (int): The number of files to load.

    Returns:
        pd.DataFrame: A combined DataFrame containing data from all files.
    """
    dataframes = []
    for i in range(num_files):
        filepath = f"{directory}{base_filename}{i}.csv"
        try:
            df = pd.read_csv(filepath)
            dataframes.append(df)
        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")

    return pd.concat(dataframes, ignore_index=True)

def clean_and_label_data(df, le_state=None, le_category=None):
    """
    Clean data by processing text columns and optionally transforming label encoders.

    Parameters:
        df (pd.DataFrame): The input data frame.
        le_state (LabelEncoder, optional): The label encoder for the 'state' column.
        le_category (LabelEncoder, optional): The label encoder for the 'mainCategory' column.

    Returns:
        tuple: Processed DataFrame, fitted/used label encoders.
    """
    if le_state is None:
        le_state = LabelEncoder()
        le_state.fit(df['pickupState'])
    if le_category is None:
        le_category = LabelEncoder()
        le_category.fit(df['mainCategory'])

    # Filter out invalid prices and missing image URLs
    df = df[
        (df['currentPrice'] > 0) &
        (df['imageUrls'].notna()) &
        (df['imageUrls'] != '')
    ]

    # Clean text columns
    df['title'] = df['title'].apply(lambda x: str(x).lower().strip() if pd.notna(x) else '')
    df['description'] = df['description'].apply(lambda x: str(x).lower().strip() if pd.notna(x) else '')
    df['mainCategory'] = df['mainCategory'].apply(lambda x: str(x).lower().strip() if pd.notna(x) else '')

    # Transform label encoders
    if le_state is not None:
        df['state_encoded'] = le_state.transform(df['pickupState'])
    if le_category is not None:
        df['category_encoded'] = le_category.transform(df['mainCategory'])

    return df, le_state, le_category

def split_data(df, test_size=100, random_state=42):
    """
    Split the data into training/validation and test sets.

    Parameters:
        df (pd.DataFrame): The input data frame.
        test_size (int): The size of the test set.
        random_state (int): The random seed for reproducibility.

    Returns:
        tuple: Training/validation DataFrame, Test DataFrame.
    """

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )
    return train_val_df, test_df