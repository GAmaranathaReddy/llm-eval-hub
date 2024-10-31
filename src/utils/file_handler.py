
import csv
import os
from datetime import datetime

def save_results_to_csv(all_results, data_dir="data/results"):
    # Ensure the results directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Create a file name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"results_{timestamp}.csv"
    filepath = os.path.join(data_dir, filename)

    # Write results to a CSV file
    with open(filepath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            'LLM Name',
            'Input Text',
            'Ground Truths',
            'Generated Output',
            'ROUGE Score',
            'BLEU Score',
            'Cosine Similarity',
            'Semantic Search Match Index',
            'Semantic Search Score'
        ])

        for result in all_results:
            for llm_result in result['llm_results']:
                metrics = llm_result['evaluation_metrics']
                writer.writerow([
                    llm_result['model'],
                    result['input_text'],
                    "; ".join(result['ground_truths']),  # Join multiple ground truths
                    llm_result['generated_output'],
                    metrics['ROUGE']['rouge1'],  # Adjust based on actual metric keys
                    metrics['BLEU']['bleu'],
                    metrics['Cosine Similarity'],
                    metrics['Semantic Search Best Match Index'],
                    metrics['Semantic Search Best Match Score']
                ])

    return filepath
