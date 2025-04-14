import csv
import os
from collections import defaultdict

def count_actor_appearances(filename: str) -> dict[str, int]:
    actor_counts = defaultdict(int)

    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            for actor_col in ['actor_1_name', 'actor_2_name', 'actor_3_name']:
                actor_name = row.get(actor_col, '').strip()
                if actor_name:
                    actor_counts[actor_name] += 1

    return dict(actor_counts)

def save_to_csv(actor_counts: dict[str, int], output_filename: str) -> None:
    # Ensure the 'data' directory exists
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    with open(output_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['actor_name', 'num_movies'])  # Header
        for actor, count in sorted(actor_counts.items(), key=lambda x: x[1], reverse=True):
            writer.writerow([actor, count])

def main():
    input_file = 'data/movie_data.csv'
    output_file = os.path.join('data', 'actor_movie_counts.csv')

    actor_movie_counts = count_actor_appearances(input_file)

    # Export to CSV in 'data' folder
    save_to_csv(actor_movie_counts, output_file)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
