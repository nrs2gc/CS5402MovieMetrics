'''model_training_with_comments.py

This script loads movie data, preprocesses it, engineers features, trains various regression models
for predicting IMDb scores, evaluates performance, analyzes actor importance and performance,
and saves the trained model for future predictions.'''

import pandas as pd  # data manipulation
import numpy as np   # numerical operations
import joblib        # model serialization
import os            # filesystem operations
import matplotlib.pyplot as plt  # plotting

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data(file_path):
    #Load dataset from a CSV file and return as a pandas DataFrame
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def preprocess_data(df):
    #Clean and preprocess raw DataFrame:
    # Drop unwanted columns
    # Impute numeric missing values with median
    # Fill categorical missing values with 'Unknown'
    # Strip whitespace from movie titles
    df_clean = df.copy()

    # Remove columns that are not needed for modeling
    columns_to_drop = ['index', 'movie_imdb_link']
    df_clean = df_clean.drop(columns=columns_to_drop, errors='ignore')

    # Impute numeric columns
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        # Replace NaNs with column median
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Impute categorical columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        # Fill missing text fields with 'Unknown'
        df_clean[col] = df_clean[col].fillna('Unknown')

    # Clean up movie titles by removing extra spaces
    if 'movie_title' in df_clean.columns:
        df_clean['movie_title'] = df_clean['movie_title'].str.strip()

    return df_clean


class ActorScoreCalculator:
    #Computes actor-based features:
    # Average IMDb score per actor
    # Total movie count per actor
    # Genre versatility per actor
    def __init__(self):
        # Dictionaries to store actor statistics
        self.actor_ratings = {}
        self.actor_movie_counts = {}
        self.actor_genres = {}
        self.trained = False

    def fit(self, df):
        # Train the scoring system by aggregating IMDb scores and genres per actor.
        required_cols = ['actor_1_name', 'actor_2_name', 'actor_3_name', 'imdb_score', 'genres']
        # Ensure all required columns are present
        if not all(col in df.columns for col in required_cols):
            raise ValueError("DataFrame is missing required columns for actor scoring")

        # Reset storage
        self.actor_ratings.clear()
        self.actor_movie_counts.clear()
        self.actor_genres.clear()

        # Iterate through each movie record
        for _, row in df.iterrows():
            score = row['imdb_score']
            # Split genre string into list
            genres = row['genres'].split('|') if isinstance(row['genres'], str) else []

            # Process each actor column
            for actor_col in ['actor_1_name', 'actor_2_name', 'actor_3_name']:
                actor = row[actor_col]
                # Skip missing/unknown actors
                if pd.isna(actor) or actor == 'Unknown':
                    continue

                if actor not in self.actor_ratings:
                    # Initialize actor entry
                    self.actor_ratings[actor] = score
                    self.actor_movie_counts[actor] = 1
                    self.actor_genres[actor] = set(genres)
                else:
                    # Update running average for actor score
                    count = self.actor_movie_counts[actor]
                    current_avg = self.actor_ratings[actor]
                    new_avg = (current_avg * count + score) / (count + 1)
                    self.actor_ratings[actor] = new_avg
                    self.actor_movie_counts[actor] += 1
                    # Accumulate unique genres
                    self.actor_genres[actor].update(genres)

        # Convert genre sets to lists for serialization
        for actor in self.actor_genres:
            self.actor_genres[actor] = list(self.actor_genres[actor])

        print(f"Actor scoring system trained on {len(self.actor_ratings)} unique actors")
        self.trained = True
        return self

    def get_actor_score(self, actor_name):
        # Return the average IMDb score for a given actor.
        if not self.trained:
            raise ValueError("Actor scoring system not trained yet")
        if pd.isna(actor_name) or actor_name == 'Unknown':
            return 0.0
        return self.actor_ratings.get(actor_name, 0.0)

    def get_actor_movie_count(self, actor_name):
        # Return the total number of movies an actor has appeared in.
        if not self.trained:
            raise ValueError("Actor scoring system not trained yet")
        if pd.isna(actor_name) or actor_name == 'Unknown':
            return 0
        return self.actor_movie_counts.get(actor_name, 0)

    def get_actor_genre_versatility(self, actor_name):
        # Return the number of unique genres an actor has worked in.
        if not self.trained:
            raise ValueError("Actor scoring system not trained yet")
        if pd.isna(actor_name) or actor_name == 'Unknown':
            return 0
        return len(self.actor_genres.get(actor_name, []))

    def transform(self, df):
        # Generate actor-based features for each movie in the DataFrame.
        if not self.trained:
            raise ValueError("Actor scoring system not trained yet")

        result = pd.DataFrame(index=df.index)

        # Create features for each actor column
        for actor_col in ['actor_1_name', 'actor_2_name', 'actor_3_name']:
            if actor_col not in df.columns:
                continue
            result[f'{actor_col}_avg_rating'] = df[actor_col].apply(self.get_actor_score)
            result[f'{actor_col}_movie_count'] = df[actor_col].apply(self.get_actor_movie_count)
            result[f'{actor_col}_genre_versatility'] = df[actor_col].apply(self.get_actor_genre_versatility)

        # Aggregate cast-level features if all actor columns are present
        cols = ['actor_1_name', 'actor_2_name', 'actor_3_name']
        if all(col in df.columns for col in cols):
            result['cast_avg_rating'] = result[[f'{c}_avg_rating' for c in cols]].mean(axis=1)
            result['cast_total_movies'] = result[[f'{c}_movie_count' for c in cols]].sum(axis=1)
            result['cast_max_rating'] = result[[f'{c}_avg_rating' for c in cols]].max(axis=1)
            # Star power combines average rating and experience
            result['cast_star_power'] = result['cast_avg_rating'] * np.log1p(result['cast_total_movies'])

        return result

    def fit_transform(self, df):
        # Convenience method: fit then transform in one step.
        return self.fit(df).transform(df)


class MovieFeatureTransformer:
    # Engineers model features from raw movie DataFrame:
    # One-hot encode genres and countries
    # Label encode directors
    # Vectorize cast names
    # Generate actor-scoring features
    # Scale numeric variables
    def __init__(self):
        self.mlb = MultiLabelBinarizer()           # for genres
        self.le_director = LabelEncoder()          # for director names
        self.actors_vectorizer = CountVectorizer(max_features=100)  # for cast text
        self.actor_scorer = ActorScoreCalculator() # actor-based features
        self.feature_names = None
        self.trained = False
        self.known_countries = None
        self.known_languages = None

    def fit(self, df):
        # Fit all internal transformers on training data.
        data = df.copy()
        # Prepare genre lists
        data['genres_list'] = data['genres'].str.split('|')
        self.mlb.fit(data['genres_list'])
        # Fit director encoder
        self.le_director.fit(data['director_name'].fillna('Unknown'))
        # Vectorize actor text
        data['all_actors'] = (data['actor_1_name'].fillna('') + ' ' +
                              data['actor_2_name'].fillna('') + ' ' +
                              data['actor_3_name'].fillna(''))
        self.actors_vectorizer.fit(data['all_actors'])
        # Train actor scorer
        self.actor_scorer.fit(data)
        # Record known categories
        self.known_countries = data['country'].fillna('Unknown').unique()
        self.known_languages = data['language'].fillna('Unknown').unique()

        self.trained = True
        return self

    def transform(self, df):
        # Transform new data into feature matrix using fitted transformers.
        if not self.trained:
            raise ValueError("Transformer has not been fitted yet")

        data = df.copy()
        # ---------- Genre features ----------
        if 'genres_list' not in data.columns:
            data['genres_list'] = data['genres'].str.split('|')
        genre_df = pd.DataFrame(
            self.mlb.transform(data['genres_list']),
            columns=self.mlb.classes_,
            index=data.index
        )

        # ---------- Director feature ----------
        data['director_name'] = data['director_name'].fillna('Unknown')
        known_dirs = set(self.le_director.classes_)
        data['director_name'] = data['director_name'].apply(lambda x: x if x in known_dirs else 'Unknown')
        data['director_encoded'] = self.le_director.transform(data['director_name'])

        # ---------- Actor vector features ----------
        data['all_actors'] = (data['actor_1_name'].fillna('') + ' ' +
                              data['actor_2_name'].fillna('') + ' ' +
                              data['actor_3_name'].fillna(''))
        actors_df = pd.DataFrame(
            self.actors_vectorizer.transform(data['all_actors']).toarray(),
            columns=self.actors_vectorizer.get_feature_names_out(),
            index=data.index
        )

        # ---------- Actor scoring features ----------
        actor_scoring_df = self.actor_scorer.transform(data)

        # ---------- Numeric features ----------
        numeric_cols = ['duration', 'num_voted_users', 'num_user_for_reviews', 'title_year']
        numeric_feats = data[[c for c in numeric_cols if c in data.columns]]

        # ---------- Country & Language one-hot ----------
        # Handle unseen categories by mapping to 'Unknown'
        data['country'] = data['country'].apply(lambda x: x if x in self.known_countries else 'Unknown')
        country_df = pd.get_dummies(data['country'], prefix='country')
        # Ensure consistent columns
        for c in self.known_countries:
            col = f'country_{c}'
            if col not in country_df:
                country_df[col] = 0

        data['language'] = data['language'].apply(lambda x: x if x in self.known_languages else 'Unknown')
        language_df = pd.get_dummies(data['language'], prefix='language')
        for l in self.known_languages:
            col = f'language_{l}'
            if col not in language_df:
                language_df[col] = 0

        # ---------- Additional engineered features ----------
        extra = pd.DataFrame(index=data.index)
        # Log-transform Facebook likes if available
        if 'actor_1_facebook_likes' in data.columns:
            extra['actor_1_facebook_likes_log'] = np.log1p(data['actor_1_facebook_likes'])
        else:
            extra['actor_1_facebook_likes_log'] = 0
        # Budget per minute of film
        if 'budget' in data.columns:
            extra['budget_per_minute'] = data['budget'] / data['duration'].replace(0, 1)
        else:
            extra['budget_per_minute'] = 0

        # ---------- Combine all features ----------
        combined = pd.concat([
            numeric_feats,
            genre_df,
            country_df,
            language_df,
            data[['director_encoded']],
            actors_df,
            actor_scoring_df,
            extra
        ], axis=1)

        # If first transform, record feature names order
        if self.feature_names is None:
            self.feature_names = combined.columns.tolist()
            return combined
        # Otherwise, align to recorded feature order
        for feat in self.feature_names:
            if feat not in combined.columns:
                combined[feat] = 0
        return combined[self.feature_names]

    def fit_transform(self, df):
        # Convenience method to fit and transform in one call.
        return self.fit(df).transform(df)

    def get_feature_names(self):
        # Get list of feature names after fitting.
        return self.feature_names


def analyze_actor_importance(model, feature_names, output_path='actor_importance.png'):
    # Plot and return the top 15 actor-related feature importances from trained model.
    # Unpack model if it's wrapped in a pipeline
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        inst = model.named_steps['model']
    else:
        inst = model

    if not hasattr(inst, 'feature_importances_'):
        print("Model doesn't provide feature importance. Skipping actor importance analysis.")
        return None

    # Gather importances into DataFrame
    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': inst.feature_importances_})
    # Filter for actor-related features
    actor_feats = imp_df[imp_df['Feature'].str.contains('actor') | imp_df['Feature'].str.contains('cast')]

    if actor_feats.empty:
        print("No actor-related features found")
        return None

    # Select top 15 actor features
    top15 = actor_feats.nlargest(15, 'Importance')

    # Create horizontal bar chart
    plt.figure(figsize=(12, 8))
    plt.barh(top15['Feature'], top15['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Actor Feature')
    plt.title('Actor Feature Importance in Movie Rating Prediction')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print("\nTop Actor Features by Importance:")
    for _, row in top15.head(10).iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")

    return top15


def train_model(X_train, y_train):
    # Train multiple regression models with GridSearchCV, select the best, and return a fitted pipeline.
    scaler = StandardScaler()  # normalize features

    # Define candidate models and hyperparameter grids
    candidates = {
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'ElasticNet': ElasticNet(random_state=42)
    }
    grids = {
        'RandomForest': {'n_estimators': [100, 200], 'max_depth': [None, 15, 30], 'min_samples_split': [2, 5]},
        'GradientBoosting': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]},
        'ElasticNet': {'alpha': [0.1, 0.5, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]}
    }

    best_score = -np.inf
    best_model = None
    best_name = None

    print("\nTraining and evaluating models...")
    # Scale training data once
    X_scaled = scaler.fit_transform(X_train)

    for name, model in candidates.items():
        print(f"\nTraining {name}...")
        gs = GridSearchCV(
            model, grids[name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1
        )
        gs.fit(X_scaled, y_train)

        score = gs.best_score_
        print(f"{name} best CV score: {-score:.4f} MSE, {np.sqrt(-score):.4f} RMSE")
        print(f"Best parameters: {gs.best_params_}")

        if score > best_score:
            best_score = score
            best_model = gs.best_estimator_
            best_name = name

    print(f"\nBest model: {best_name} with RMSE: {np.sqrt(-best_score):.4f}")
    # Create final pipeline
    pipeline = Pipeline([('scaler', scaler), ('model', best_model)])
    pipeline.fit(X_train, y_train)
    return pipeline, best_name


def evaluate_model(model, X_test, y_test):
    # Evaluate the trained model on test data, produce plots and return metrics.
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

    # Scatter plot: actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.title('Actual vs Predicted Movie Ratings')
    plt.savefig('prediction_results.png')
    plt.close()

    # Error distribution histogram
    errors = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=20, alpha=0.6)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.savefig('error_distribution.png')
    plt.close()

    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


def save_model(model, feature_transformer, output_dir='model'):
    # Persist model and transformer to disk using joblib.
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
    # Load model and transformer from disk.
    try:
        model = joblib.load(os.path.join(model_dir, 'movie_rating_model.pkl'))
        transformer = joblib.load(os.path.join(model_dir, 'feature_transformer.pkl'))
        print("Model loaded successfully")
        return model, transformer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def predict_rating(new_movie_data_path, model_dir='model'):
    # Generate and save predicted ratings for new movies provided in CSV.
    model, transformer = load_model(model_dir)
    if model is None:
        return None

    df_new = load_data(new_movie_data_path)
    if df_new is None:
        return None

    # Ensure required columns exist
    required = ['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name',
                'genres', 'duration', 'title_year', 'country', 'language',
                'num_voted_users', 'num_user_for_reviews']
    missing = [c for c in required if c not in df_new.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        return None

    preproc = preprocess_data(df_new)
    try:
        X_new = transformer.transform(preproc)
        print(f"Transformed features shape: {X_new.shape}")
    except Exception as e:
        print(f"Error transforming features: {e}")
        return None

    preds = model.predict(X_new)
    df_res = df_new.copy()
    df_res['predicted_rating'] = preds

    # Add actor average ratings for context
    df_res['actor1_avg_rating'] = preproc['actor_1_name'].apply(transformer.actor_scorer.get_actor_score)
    df_res['actor2_avg_rating'] = preproc['actor_2_name'].apply(transformer.actor_scorer.get_actor_score)
    df_res['actor3_avg_rating'] = preproc['actor_3_name'].apply(transformer.actor_scorer.get_actor_score)

    # Save predictions to CSV
    out_path = 'predicted_movie_ratings.csv'
    df_res.to_csv(out_path, index=False)
    print(f"\nPredictions saved to {out_path}")

    # Print summary for each movie
    print("\nPrediction Summary:")
    for i, row in df_res.iterrows():
        title = row.get('movie_title', f'Movie {i+1}')
        print(f"Movie: {title} - Predicted Rating: {row['predicted_rating']:.2f}/10")
        print(f"  Lead Actor: {row.get('actor_1_name', 'Unknown')} "
              f"(Avg Rating: {row['actor1_avg_rating']:.2f})")

    return df_res


def analyze_actor_performance(df, actor_scorer):
    # Visualize top actors by rating and versatility based on training data.
    actor_stats = []
    # Collect data for actors with at least 3 movies
    for actor, count in actor_scorer.actor_movie_counts.items():
        if count >= 3:
            actor_stats.append({
                'Actor': actor,
                'Average Rating': actor_scorer.actor_ratings[actor],
                'Movie Count': count,
                'Genre Count': len(actor_scorer.actor_genres[actor])
            })

    if not actor_stats:
        print("No actors with sufficient data for analysis")
        return None

    stats_df = pd.DataFrame(actor_stats)
    # Top 20 by average rating
    top_rated = stats_df.nlargest(20, 'Average Rating')

    # Plot top-rated actors
    plt.figure(figsize=(12, 8))
    plt.barh(top_rated['Actor'], top_rated['Average Rating'])
    plt.xlabel('Average Movie Rating')
    plt.title('Top 20 Actors by Average Movie Rating')
    plt.xlim(0, 10)
    plt.tight_layout()
    plt.savefig('top_actors.png')
    plt.close()

    # Top 20 by genre versatility
    top_versatile = stats_df.sort_values(['Genre Count', 'Average Rating'], ascending=False).head(20)
    plt.figure(figsize=(12, 8))
    plt.barh(top_versatile['Actor'], top_versatile['Genre Count'])
    plt.xlabel('Number of Different Genres')
    plt.title('Most Versatile Actors by Number of Genres')
    plt.tight_layout()
    plt.savefig('versatile_actors.png')
    plt.close()

    print("\nActor Performance Analysis:")
    for _, row in top_rated.head(5).iterrows():
        print(f"{row['Actor']}: {row['Average Rating']:.2f} (from {row['Movie Count']} movies)")
    for _, row in top_versatile.head(5).iterrows():
        print(f"{row['Actor']}: {row['Genre Count']} genres (Rating: {row['Average Rating']:.2f})")

    return stats_df


def main(file_path, test_new_movie_path=None):
    # Main workflow: load data, preprocess, feature engineering, train/evaluate model, save artifacts.
    df = load_data(file_path)
    if df is None:
        return

    print("\nInitial data overview:")
    print(df.info())

    df_clean = preprocess_data(df)
    print(f"\nMissing values after cleaning: {df_clean.isnull().sum().sum()}")

    # Feature engineering
    feat_transformer = MovieFeatureTransformer()
    features = feat_transformer.fit_transform(df_clean)
    print(f"\nEngineered features shape: {features.shape}")

    target = df_clean['imdb_score']
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

    # Train and select model
    model, name = train_model(X_train, y_train)
    # Evaluate performance
    metrics = evaluate_model(model, X_test, y_test)
    # Analyze actor-related features
    if name in ['RandomForest', 'GradientBoosting']:
        analyze_actor_importance(model, features.columns)
    analyze_actor_performance(df_clean, feat_transformer.actor_scorer)

    # Save trained artifacts
    save_model(model, feat_transformer)

    # Optionally predict on new data
    if test_new_movie_path:
        print("\nMaking predictions for new movies...")
        predict_rating(test_new_movie_path)

    return model, feat_transformer, metrics


if __name__ == "__main__":
    # Define file paths
    train_file_path = "data/movie_data.csv"
    new_movie_file_path = "data/new_movies.csv"
    # Execute main workflow
    main(train_file_path, new_movie_file_path)
