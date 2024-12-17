
import pickle
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

with open("best_scores.pkl", "rb") as file:  # Open the file in read-binary mode
    loaded_data = pickle.load(file)
print(loaded_data)

def process_data(data, file_name="sorted_data.csv"):
    """
    Converts a list of dictionaries into a pandas DataFrame, sorts it by accuracy, 
    and saves the DataFrame to a CSV file.

    Parameters:
        data (list): A list of dictionaries containing 'scalar', 'lr', 'drop', and 'accuracy'.
        file_name (str): The name of the file to save the DataFrame to.

    Returns:
        pd.DataFrame: The sorted DataFrame.
    """
    # Convert list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    
    # Sort the DataFrame by 'accuracy' in descending order
    df_sorted = df.sort_values(by="scalar", ascending=False)
    
    # Save the sorted DataFrame to a CSV file
    df_sorted.to_csv(file_name, index=False)
    
    return df_sorted

sorted_df = process_data(loaded_data)



"""goes through all scalar values and average the accuracy"""
def average_scalar_accuracy():
    the_sums = defaultdict(float) 
    occurences = Counter()

    for datum in loaded_data:
        the_scalar = datum["scalar"]
        the_sums[the_scalar] += datum["accuracy"]
        occurences[the_scalar] +=1

    averages = defaultdict(float)
    
    for key in the_sums.keys():
        averages[key] = the_sums[key] / occurences[key]

    return averages

averages = average_scalar_accuracy()
#defaultdict(<class 'float'>, {0.5: 0.7916926312609595, 0.7: 0.7860403582688614, 0.9: 0.7722740738011488, 1.0: 0.7893817197405696, 1.1: 0.7958545444467883, 1.3: 0.7994522408355372})

def plot():
    xvalues, yvalues = [], []
    for scalar, accuracy in averages.items():
        xvalues.append(scalar)
        yvalues.append(accuracy)
    plt.plot(xvalues, yvalues, label="average accuracy by scalar")
    plt.xlabel("scalar")  # Add X-axis label
    plt.ylabel("accuracy")  # Add Y-axis label
    plt.title("average accuracy by scalar")  # Add a title
    plt.legend()  # Add a legend
    plt.grid(True)  # Add gridlines
    plt.show()  # Display the plot

plot()