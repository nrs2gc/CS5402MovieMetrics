import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def preprocess_data(df):
    df_clean = df.copy()
    
    columns_to_drop = ['index', 'movie_imdb_link']
    df_clean = df_clean.drop(columns=columns_to_drop, errors='ignore')
    
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna('Unknown')
    
    if 'movie_title' in df_clean.columns:
        df_clean['movie_title'] = df_clean['movie_title'].str.strip()
    
    return df_clean

def engineer_features(df):
    df['genres_list'] = df['genres'].str.split('|')
    
    mlb = MultiLabelBinarizer()
    genre_features = pd.DataFrame(mlb.fit_transform(df['genres_list']),
                                 columns=mlb.classes_,
                                 index=df.index)
    
    le_director = LabelEncoder()
    df['director_encoded'] = le_director.fit_transform(df['director_name'])
    
    df['all_actors'] = df['actor_1_name'] + ' ' + df['actor_2_name'] + ' ' + df['actor_3_name']
    
    actors_vectorizer = CountVectorizer(max_features=100)
    actors_features = pd.DataFrame(
        actors_vectorizer.fit_transform(df['all_actors']).toarray(),
        columns=actors_vectorizer.get_feature_names_out(),
        index=df.index
    )
    
    numeric_features = df[['duration', 'num_voted_users', 'num_user_for_reviews', 'title_year']]
    
    combined_features = pd.concat([
        numeric_features, 
        genre_features, 
        pd.get_dummies(df[['country', 'language']]), 
        pd.DataFrame(df['director_encoded'], index=df.index),
        actors_features
    ], axis=1)
    
    return combined_features

def prepare_for_model(features_df, target_series, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, 
        target_series, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def main(file_path):
    df = load_data(file_path)
    if df is None:
        return
    
    print("\nInitial data overview:")
    print(df.info())
    print("\nSample data:")
    print(df.head())
    
    df_clean = preprocess_data(df)
    print("\nData after cleaning:")
    print(df_clean.isnull().sum().sum(), "missing values remain")
    
    features_df = engineer_features(df_clean)
    print("\nEngineered features shape:", features_df.shape)
    
    target = df_clean['imdb_score']
    
    X_train, X_test, y_train, y_test = prepare_for_model(features_df, target)
    
    print("\nPreprocessing complete")
    return X_train, X_test, y_train, y_test, features_df.columns

if __name__ == "__main__":
    file_path = "data/movie_data.csv"  
    main(file_path)
  
