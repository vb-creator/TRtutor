import json
import numpy as np
import seaborn as sns
from collections import Counter
from statistics import mode, StatisticsError
import os

import matplotlib.pyplot as plt

# Path to your .jsonl file
file_path = 'llm_eval_results.jsonl'

# Load the data
def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Extract scores for each type
def extract_scores(data):
    scores = {
        "coherence": [],
        "relevance": [],
        "personalization": [],
        "engagement": [],
        "instructional_quality": []
    }
    for entry in data:
        for score_type in scores.keys():
            scores[score_type].append(entry[score_type]["score"])
    return scores

# Compute mean, median, and mode
def compute_statistics(scores):
    stats = {}
    for score_type, values in scores.items():
        mean = np.mean(values)
        median = np.median(values)
        try:
            mode_value = mode(values)
        except StatisticsError:
            mode_value = "No unique mode"
        stats[score_type] = {"mean": mean, "median": median, "mode": mode_value}
    return stats

# Plot density graphs
def plot_density(scores):
    # Create the plots directory if it doesn't exist
    os.makedirs('./plots', exist_ok=True)
    
    for score_type, values in scores.items():
        sns.kdeplot(values, fill=True)
        plt.title(f'Density Plot for {score_type.capitalize()}')
        plt.xlabel('Score')
        plt.ylabel('Density')
        
        # Save the plot
        plot_path = f'./plots/{score_type}_density_plot.png'
        plt.savefig(plot_path)
        plt.close()  # Close the plot to avoid overlapping
        
        print(f"Saved plot for {score_type} at {plot_path}")

# Main function
def main():
    data = load_data(file_path)
    scores = extract_scores(data)
    stats = compute_statistics(scores)
    
    # Print statistics
    for score_type, stat in stats.items():
        print(f"{score_type.capitalize()} Statistics:")
        print(f"  Mean: {stat['mean']}")
        print(f"  Median: {stat['median']}")
        print(f"  Mode: {stat['mode']}")
        print()
    
    # Plot density graphs
    plot_density(scores)

if __name__ == "__main__":
    main()