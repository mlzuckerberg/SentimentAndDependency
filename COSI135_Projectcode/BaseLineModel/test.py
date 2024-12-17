import spacy
from spacy.training import Example
from typing import Iterable, List, Tuple, Dict, Set

nlp = spacy.load("en_core_web_md") 


def extract_dependency_layers(sentence: str) -> List[List[str]]:
    """
    Extract dependency layers from a sentence by following dependency tree arrows.
    
    Args:
        sentence (str): Input sentence to parse
    
    Returns:
        List of lists, where each inner list represents words reached by following 
        dependency tree arrows from the root
    """
    # Load spaCy model
   
    
    # Process the sentence
    doc = nlp(sentence)
    
    # Find the main verb (root) of the sentence
    main_verb = None
    for token in doc:
        if token.dep_ == "ROOT":
            main_verb = token
            break
    
    if main_verb is None:
        return []
    
    # Initialize layers dictionary
    layers: Dict[int, Set[str]] = {0: {main_verb.text}}
    
    # Track visited tokens to prevent infinite loops
    visited = set()
    
    def trace_dependency_layers(current_token, current_depth=0):
        # Mark current token as visited
        visited.add(current_token)
        
        # Explore children
        for child in current_token.children:
            # Skip if already visited
            if child in visited:
                continue
            
            # Add to appropriate layer
            depth = current_depth + 1
            if depth not in layers:
                layers[depth] = set()
            layers[depth].add(child.text)
            
            # Recursively explore this child's dependencies
            trace_dependency_layers(child, depth)
        
        # Explore parent if not already at root
        if current_token.head != current_token and current_token.head not in visited:
            depth = current_depth + 1
            if depth not in layers:
                layers[depth] = set()
            layers[depth].add(current_token.head.text)
            trace_dependency_layers(current_token.head, depth)
    
    # Start tracing from main verb
    trace_dependency_layers(main_verb)
    
    # Convert to sorted list of layers, converting sets to lists
    max_layer = max(layers.keys()) if layers else 0
    result = [list(layers.get(i, set())) for i in range(max_layer + 1)]
    
    return result

sentence = "the quick fat cat ran slowly to his mother"
print(extract_dependency_layers(sentence))


def weight_words(dep_layers: Iterable[Iterable[str]], k: float):
    """
    Parameters:
    dep_layers (Iterable[Iterable[str]]): A list of lists containing words sorted by dependency to modify.
    k (float): Initial scaling factor for word vectors. Decrease withd istance from main verb
    """
    curr_k = k
    iteration_count = 1

    for layer in dep_layers:
        for word in layer:
            # Check if the word exists in the vocabulary
            if word in nlp.vocab:
                # Scale the vector by curr_k
                nlp.vocab[word].vector = nlp.vocab[word].vector * curr_k
            else:
                print(f"Word '{word}' not found in the vocabulary.")
        
        # Update scaling factor
        iteration_count += 1
        curr_k = k / iteration_count  # Recalculate curr_k

    print("Word vectors updated successfully.")


test_sentence = "The quick brown iguana slither around the school with astonishing speed"

layers = extract_dependency_layers(test_sentence)

weight_words(layers, 16)