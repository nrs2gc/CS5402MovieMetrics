import csv  # to read and write CSV files
import os   # to handle filesystem operations
from collections import defaultdict  # to create dictionaries with default values

def count_actor_appearances_with_scores(filename: str) -> dict[str, tuple[int, float]]:
    """
    Reads a CSV of movie data and computes, for each actor:
    - the number of movies they've appeared in
    - their average movie score (with a default of 5.00 for actors with fewer than 5 appearances)
    """
    # Counters for number of movies per actor
    actor_movie_counts = defaultdict(int)
    # Accumulators for total IMDB scores per actor
    actor_total_scores = defaultdict(float)

    # Open the input CSV file
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        # Process each row/movie in the CSV
        for row in reader:
            # Parse the score; skip if invalid
            try:
                score = float(row.get('imdb_score', '').strip())
            except (ValueError, TypeError):
                score = None  # Use None for missing or malformed scores

            # Iterate through the actor columns in the row
            for actor_col in ['actor_1_name', 'actor_2_name', 'actor_3_name']:
                actor_name = row.get(actor_col, '').strip()
                if actor_name:
                    # Count this movie for the actor
                    actor_movie_counts[actor_name] += 1
                    # Accumulate score only if valid
                    if score is not None:
                        actor_total_scores[actor_name] += score

    # Prepare the final output dictionary
    actor_data = {}
    for actor, count in actor_movie_counts.items():
        total_score = actor_total_scores.get(actor, 0.0)
        # Apply default average score for actors with fewer than 5 movies
        if count < 5:
            average_score = 5.00
        else:
            average_score = round(total_score / count, 2)
        actor_data[actor] = (count, average_score)

    return actor_data

def save_to_csv(actor_data: dict[str, tuple[int, float]], output_filename: str) -> None:
    # Saves the actor data to a CSV file sorted by descending number of movies.
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # Write to the CSV file
    with open(output_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['actor_name', 'num_movies', 'actor_score'])  # Header row
        # Sort actors by movie count descending for readability
        for actor, (count, avg_score) in sorted(
            actor_data.items(), key=lambda x: x[1][0], reverse=True
        ):
            writer.writerow([actor, count, avg_score])

def main():
    # - Defines input and output file paths
    # - Computes actor appearance statistics
    # - Saves results to a CSV
    input_file = 'data/movie_data.csv'
    output_file = os.path.join('data', 'actor_movie_counts.csv')

    # Compute actor data from input CSV
    actor_data = count_actor_appearances_with_scores(input_file)

    # Save the computed data to output CSV
    save_to_csv(actor_data, output_file)
    print(f"\nResults saved to: {output_file}")

# Standard boilerplate to run the main function
if __name__ == "__main__":
    main()
