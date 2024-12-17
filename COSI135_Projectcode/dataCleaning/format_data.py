
import re
from typing import List, Tuple

def process_kaggle(file_path: str) -> List[Tuple[str, dict[str, dict[str, float]]]]:
    """
    Process a text file with sentences and binary sentiment labels.
    
    Args:
        file_path (str): Path to the input text file
    
    Returns:
        List of tuples, each containing:
        - Lowercase processed sentence
        - Sentiment dictionary with POSITIVE and NEGATIVE probabilities
    """
    processed_data = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Split the line into text and label
            parts = line.strip().split('\t')
            
            # Handle cases with unexpected line format
            if len(parts) != 2:
                continue
            
            # Extract text and label
            text, label = parts[0], parts[1]
            
            # Preprocess text: lowercase, remove extra whitespace
            cleaned_text = re.sub(r'\s+', ' ', text.lower()).strip()
            
            # Create sentiment dictionary based on label
            sentiment = {
                "cats": {
                    "POSITIVE": float(label),
                    "NEGATIVE": 1.0 - float(label)
                }
            }
            
            processed_data.append((cleaned_text, sentiment))
    #return processed_data
    



process_kaggle('kaggle_og.txt')