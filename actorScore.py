import csv
import os
from collections import defaultdict

def count_actor_appearances_with_scores(filename: str) -> dict[str, tuple[int, float]]:
    actor_movie_counts = defaultdict(int)
    actor_total_scores = defaultdict(float)

    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            try:
                score = float(row.get('imdb_score', '').strip())
            except (ValueError, TypeError):
                score = None  # Skip invalid scores

            for actor_col in ['actor_1_name', 'actor_2_name', 'actor_3_name']:
                actor_name = row.get(actor_col, '').strip()
                if actor_name:
                    actor_movie_counts[actor_name] += 1
                    if score is not None:
                        actor_total_scores[actor_name] += score

    # Combine into a dict: actor -> (num_movies, actor_score)
    actor_data = {}
    for actor in actor_movie_counts:
        count = actor_movie_counts[actor]
        total_score = actor_total_scores.get(actor, 0.0)
        if count < 5:
            average_score = 5.00
        else:
            average_score = round(total_score / count, 2) if count > 0 else 5.00
        actor_data[actor] = (count, average_score)

    return actor_data

def save_to_csv(actor_data: dict[str, tuple[int, float]], output_filename: str) -> None:
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    with open(output_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['actor_name', 'num_movies', 'actor_score'])  # Header
        for actor, (count, avg_score) in sorted(actor_data.items(), key=lambda x: x[1][0], reverse=True):
            writer.writerow([actor, count, avg_score])

def main():
    input_file = 'data/movie_data.csv'
    output_file = os.path.join('data', 'actor_movie_counts.csv')

    actor_data = count_actor_appearances_with_scores(input_file)

    # Save to CSV
    save_to_csv(actor_data, output_file)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
