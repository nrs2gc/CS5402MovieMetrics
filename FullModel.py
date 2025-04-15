# Still need to work on the actor rating a bit
# takes a while to run so will try to work on that
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

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

class ActorScoreCalculator:
    def __init__(self):
        self.actor_ratings = {}
        self.actor_movie_counts = {}
        self.actor_genres = {}
        self.trained = False
    
    def fit(self, df):
        required_cols = ['actor_1_name', 'actor_2_name', 'actor_3_name', 'imdb_score', 'genres']
        if not all(col in df.columns for col in required_cols): #if all columns in required cols are not in input data frame columns raise Value Err
            raise ValueError("DataFrame is missing required columns for actor scoring")
        
        self.actor_ratings = {}
        self.actor_movie_counts = {}
        self.actor_genres = {}
        
        for _, row in df.iterrows():
            score = row['imdb_score']
            genres = row['genres'].split('|') if isinstance(row['genres'], str) else []
            
            for actor_col in ['actor_1_name', 'actor_2_name', 'actor_3_name']:
                actor = row[actor_col]
                if pd.isna(actor) or actor == 'Unknown':
                    continue
                    
                if actor not in self.actor_ratings: #create new actor entry
                    self.actor_ratings[actor] = score
                    self.actor_movie_counts[actor] = 1
                    self.actor_genres[actor] = set(genres)
                else: #increment current movie count by one, recalc actor rating avg, and update actor's genres if need be
                    current_count = self.actor_movie_counts[actor]
                    current_avg = self.actor_ratings[actor]
                    new_avg = (current_avg * current_count + score) / (current_count + 1)
                    
                    self.actor_ratings[actor] = new_avg
                    self.actor_movie_counts[actor] += 1
                    self.actor_genres[actor].update(genres)
        
        for actor in self.actor_genres:
            self.actor_genres[actor] = list(self.actor_genres[actor])
            
        print(f"Actor scoring system trained on {len(self.actor_ratings)} unique actors")
        self.trained = True
        return self
    
    def get_actor_score(self, actor_name):
        if not self.trained:
            raise ValueError("Actor scoring system not trained yet")
            
        if pd.isna(actor_name) or actor_name == 'Unknown' or actor_name not in self.actor_ratings:
            return 0.0
            
        return self.actor_ratings.get(actor_name, 0.0)
    
    def get_actor_movie_count(self, actor_name):
        if not self.trained:
            raise ValueError("Actor scoring system not trained yet")
            
        if pd.isna(actor_name) or actor_name == 'Unknown':
            return 0
            
        return self.actor_movie_counts.get(actor_name, 0)
    
    def get_actor_genre_versatility(self, actor_name):
        if not self.trained:
            raise ValueError("Actor scoring system not trained yet")
            
        if pd.isna(actor_name) or actor_name == 'Unknown':
            return 0
            
        return len(self.actor_genres.get(actor_name, []))
    
    def transform(self, df):
        if not self.trained:
            raise ValueError("Actor scoring system not trained yet")
            
        result = pd.DataFrame(index=df.index)
        
        for actor_col in ['actor_1_name', 'actor_2_name', 'actor_3_name']:
            if actor_col not in df.columns:
                continue
                
            result[f'{actor_col}_avg_rating'] = df[actor_col].apply(self.get_actor_score)
            
            result[f'{actor_col}_movie_count'] = df[actor_col].apply(self.get_actor_movie_count)
            
            result[f'{actor_col}_genre_versatility'] = df[actor_col].apply(self.get_actor_genre_versatility)
        
        if 'actor_1_name' in df.columns and 'actor_2_name' in df.columns and 'actor_3_name' in df.columns:
            result['cast_avg_rating'] = (
                result['actor_1_name_avg_rating'] + 
                result['actor_2_name_avg_rating'] + 
                result['actor_3_name_avg_rating']
            ) / 3
            
            result['cast_total_movies'] = (
                result['actor_1_name_movie_count'] + 
                result['actor_2_name_movie_count'] + 
                result['actor_3_name_movie_count']
            )
            
            result['cast_max_rating'] = result[['actor_1_name_avg_rating', 
                                               'actor_2_name_avg_rating', 
                                               'actor_3_name_avg_rating']].max(axis=1)
            
            result['cast_star_power'] = (
                result['cast_avg_rating'] * np.log1p(result['cast_total_movies'])
            )
        
        return result
    
    def fit_transform(self, df):
        return self.fit(df).transform(df)

class MovieFeatureTransformer:
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        self.le_director = LabelEncoder()
        self.actors_vectorizer = CountVectorizer(max_features=100)
        self.actor_scorer = ActorScoreCalculator()
        self.feature_names = None
        self.trained = False
        self.known_countries = None
        self.known_languages = None
        
    def fit(self, df):
        df = df.copy()
        
        df['genres_list'] = df['genres'].str.split('|')
        self.mlb.fit(df['genres_list'])
        
        self.le_director.fit(df['director_name'])
        
        df['all_actors'] = df['actor_1_name'] + ' ' + df['actor_2_name'] + ' ' + df['actor_3_name']
        self.actors_vectorizer.fit(df['all_actors'])
        
        self.actor_scorer.fit(df)
        
        self.known_countries = df['country'].unique()
        self.known_languages = df['language'].unique()
        
        self.trained = True
        return self
    
    def transform(self, df):
        if not self.trained:
            raise ValueError("Transformer has not been fitted yet")
        
        df = df.copy()
        
        if 'genres_list' not in df.columns:
            df['genres_list'] = df['genres'].str.split('|')
        genre_features = pd.DataFrame(
            self.mlb.transform(df['genres_list']),
            columns=self.mlb.classes_,
            index=df.index
        )
        
        df['director_name'] = df['director_name'].fillna('Unknown') #replace Null values
        known_directors = set(self.le_director.classes_)
        df['director_name'] = df['director_name'].apply(lambda x: x if x in known_directors else 'Unknown')
        df['director_encoded'] = self.le_director.transform(df['director_name'])
        
        df['all_actors'] = df['actor_1_name'] + ' ' + df['actor_2_name'] + ' ' + df['actor_3_name']
        actors_features = pd.DataFrame(
            self.actors_vectorizer.transform(df['all_actors']).toarray(),
            columns=self.actors_vectorizer.get_feature_names_out(),
            index=df.index
        )
        
        actor_scoring_features = self.actor_scorer.transform(df)
        
        numeric_cols = ['duration', 'num_voted_users', 'num_user_for_reviews', 'title_year']
        available_numeric_cols = [col for col in numeric_cols if col in df.columns]
        numeric_features = df[available_numeric_cols]
        
        df['country'] = df['country'].apply(lambda x: x if x in self.known_countries else 'Unknown')
        country_dummies = pd.get_dummies(df['country'], prefix='country')
        for country in self.known_countries:
            col_name = f'country_{country}'
            if col_name not in country_dummies.columns:
                country_dummies[col_name] = 0
        
        df['language'] = df['language'].apply(lambda x: x if x in self.known_languages else 'Unknown')
        language_dummies = pd.get_dummies(df['language'], prefix='language')
        for language in self.known_languages:
            col_name = f'language_{language}'
            if col_name not in language_dummies.columns:
                language_dummies[col_name] = 0
        
        additional_features = pd.DataFrame(index=df.index)
        
        if 'actor_1_facebook_likes' in df.columns:
            additional_features['actor_1_facebook_likes_log'] = np.log1p(df['actor_1_facebook_likes'])
        else:
            additional_features['actor_1_facebook_likes_log'] = 0
        
        if 'budget' in df.columns:
            additional_features['budget_per_minute'] = df['budget'] / df['duration'].replace(0, 1)
        else:
            additional_features['budget_per_minute'] = 0
        
        combined_features = pd.concat([
            numeric_features,
            genre_features,
            country_dummies,
            language_dummies, 
            pd.DataFrame(df['director_encoded'], index=df.index, columns=['director_encoded']),
            actors_features,
            actor_scoring_features,
            additional_features
        ], axis=1)
        
        if self.feature_names is None:
            self.feature_names = combined_features.columns.tolist()
            return combined_features
        else:
            for feature in self.feature_names:
                if feature not in combined_features.columns:
                    combined_features[feature] = 0
            
            return combined_features[self.feature_names]
    
    def fit_transform(self, df):
        return self.fit(df).transform(df)
    
    def get_feature_names(self):
        return self.feature_names

def analyze_actor_importance(model, feature_names, output_path='actor_importance.png'):
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        model_instance = model.named_steps['model']
    else:
        model_instance = model
    
    if not hasattr(model_instance, 'feature_importances_'):
        print("Model doesn't provide feature importance. Skipping actor importance analysis.")
        return None
    
    importances = model_instance.feature_importances_
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    actor_features = feature_importance[
        feature_importance['Feature'].str.contains('actor') | 
        feature_importance['Feature'].str.contains('cast')
    ]
    
    if actor_features.empty:
        print("No actor-related features found")
        return None
    
    actor_features = actor_features.sort_values('Importance', ascending=False).head(15)
    
    plt.figure(figsize=(12, 8))
    plt.barh(actor_features['Feature'], actor_features['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Actor Feature')
    plt.title('Actor Feature Importance in Movie Rating Prediction')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"\nTop Actor Features by Importance:")
    for i, row in actor_features.head(10).iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")
    
    return actor_features

def train_model(X_train, y_train):
    scaler = StandardScaler()
    
    models = {
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'ElasticNet': ElasticNet(random_state=42)
    }
    
    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 15, 30],
            'min_samples_split': [2, 5]
        },
        'GradientBoosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        },
        'ElasticNet': {
            'alpha': [0.1, 0.5, 1.0],
            'l1_ratio': [0.2, 0.5, 0.8]
        }
    }
    
    best_score = float('-inf')
    best_model = None
    best_model_name = None
    
    print("\nTraining and evaluating models...")
    X_train_scaled = scaler.fit_transform(X_train)
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        grid_search = GridSearchCV(
            model, 
            param_grids[model_name], 
            cv=5, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        score = grid_search.best_score_
        print(f"{model_name} best CV score: {-score:.4f} MSE, {np.sqrt(-score):.4f} RMSE")
        print(f"Best parameters: {grid_search.best_params_}")
        
        if score > best_score:
            best_score = score
            best_model = grid_search.best_estimator_
            best_model_name = model_name
    
    print(f"\nBest model: {best_model_name} with RMSE: {np.sqrt(-best_score):.4f}")
    
    pipeline = Pipeline([
        ('scaler', scaler),
        ('model', best_model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    return pipeline, best_model_name

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Evaluation:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.title('Actual vs Predicted Movie Ratings')
    plt.savefig('prediction_results.png')
    plt.close()
    
    errors = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=20, alpha=0.6)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.savefig('error_distribution.png')
    plt.close()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def save_model(model, feature_transformer, output_dir='model'):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        joblib.dump(model, os.path.join(output_dir, 'movie_rating_model.pkl'))
        joblib.dump(feature_transformer, os.path.join(output_dir, 'feature_transformer.pkl'))
        
        print(f"\nModel saved successfully to {output_dir}")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def load_model(model_dir='model'):
    try:
        model = joblib.load(os.path.join(model_dir, 'movie_rating_model.pkl'))
        feature_transformer = joblib.load(os.path.join(model_dir, 'feature_transformer.pkl'))
        
        print("Model loaded successfully")
        return model, feature_transformer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def predict_rating(new_movie_data_path, model_dir='model'):
    model, feature_transformer = load_model(model_dir)
    if model is None or feature_transformer is None:
        return None
    
    new_movie_df = load_data(new_movie_data_path)
    if new_movie_df is None:
        return None
    
    required_columns = ['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name', 
                       'genres', 'duration', 'title_year', 'country', 'language',
                       'num_voted_users', 'num_user_for_reviews']
    
    missing_columns = [col for col in required_columns if col not in new_movie_df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return None
    
    preprocessed_df = preprocess_data(new_movie_df)
    
    try:
        X_new = feature_transformer.transform(preprocessed_df)
        print(f"Transformed features shape: {X_new.shape}")
    except Exception as e:
        print(f"Error transforming features: {e}")
        print(f"Feature transformer has features: {len(feature_transformer.feature_names)}")
        return None
    
    predictions = model.predict(X_new)
    
    result_df = new_movie_df.copy()
    result_df['predicted_rating'] = predictions
    
    result_df['actor1_avg_rating'] = preprocessed_df['actor_1_name'].apply(
        feature_transformer.actor_scorer.get_actor_score)
    result_df['actor2_avg_rating'] = preprocessed_df['actor_2_name'].apply(
        feature_transformer.actor_scorer.get_actor_score)
    result_df['actor3_avg_rating'] = preprocessed_df['actor_3_name'].apply(
        feature_transformer.actor_scorer.get_actor_score)
    
    output_path = 'predicted_movie_ratings.csv'
    result_df.to_csv(output_path, index=False)
    
    print(f"\nPredictions saved to {output_path}")
    
    print("\nPrediction Summary:")
    for i, row in result_df.iterrows():
        movie_title = row.get('movie_title', f'Movie {i+1}')
        rating = row['predicted_rating']
        lead_actor = row.get('actor_1_name', 'Unknown')
        lead_actor_rating = row.get('actor1_avg_rating', 0)
        
        print(f"Movie: {movie_title} - Predicted Rating: {rating:.2f}/10")
        print(f"  Lead Actor: {lead_actor} (Avg Rating: {lead_actor_rating:.2f})")
    
    return result_df

def analyze_actor_performance(df, actor_scorer):
    actor_data = []
    
    for actor, count in actor_scorer.actor_movie_counts.items():
        if count >= 3:
            actor_data.append({
                'Actor': actor,
                'Average Rating': actor_scorer.actor_ratings[actor],
                'Movie Count': count,
                'Genre Count': len(actor_scorer.actor_genres[actor])
            })
    
    if not actor_data:
        print("No actors with sufficient data for analysis")
        return None
    
    actor_df = pd.DataFrame(actor_data)
    
    top_actors = actor_df.sort_values('Average Rating', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    ax = plt.barh(top_actors['Actor'], top_actors['Average Rating'])
    plt.xlabel('Average Movie Rating')
    plt.ylabel('Actor')
    plt.title('Top 20 Actors by Average Movie Rating')
    plt.xlim(0, 10)
    
    for i, v in enumerate(top_actors['Average Rating']):
        plt.text(v + 0.1, i, f"({top_actors['Movie Count'].iloc[i]} movies)", 
                 va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('top_actors.png')
    plt.close()
    
    versatile_actors = actor_df.sort_values(['Genre Count', 'Average Rating'], 
                                           ascending=[False, False]).head(20)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(versatile_actors['Actor'], versatile_actors['Genre Count'])
    plt.xlabel('Number of Different Genres')
    plt.ylabel('Actor')
    plt.title('Most Versatile Actors by Number of Genres')
    
    for i, v in enumerate(versatile_actors['Genre Count']):
        plt.text(v + 0.1, i, f"(Rating: {versatile_actors['Average Rating'].iloc[i]:.2f})", 
                 va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('versatile_actors.png')
    plt.close()
    
    print("\nActor Performance Analysis:")
    print(f"Top 5 Actors by Average Rating:")
    for i, row in top_actors.head().iterrows():
        print(f"{row['Actor']}: {row['Average Rating']:.2f} (from {row['Movie Count']} movies)")
    
    print(f"\nMost Versatile Actors:")
    for i, row in versatile_actors.head().iterrows():
        print(f"{row['Actor']}: {row['Genre Count']} different genres (Rating: {row['Average Rating']:.2f})")
    
    return actor_df

def main(file_path, test_new_movie_path=None):
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
    
    feature_transformer = MovieFeatureTransformer()
    features_df = feature_transformer.fit_transform(df_clean)
    print("\nEngineered features shape:", features_df.shape)
    
    target = df_clean['imdb_score']
    
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, target, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    model, model_name = train_model(X_train, y_train)
    
    metrics = evaluate_model(model, X_test, y_test)
    
    if model_name in ['RandomForest', 'GradientBoosting']:
        actor_importance = analyze_actor_importance(model, features_df.columns)
    
    actor_performance = analyze_actor_performance(df_clean, feature_transformer.actor_scorer)
    
    save_model(model, feature_transformer)
    
    if test_new_movie_path:
        print("\nMaking predictions for new movies...")
        predict_rating(test_new_movie_path)
    
    return model, feature_transformer, metrics

if __name__ == "__main__":
    train_file_path = "data/movie_data.csv"
    
    new_movie_file_path = "data/new_movies.csv"
    
    main(train_file_path, new_movie_file_path)
    
