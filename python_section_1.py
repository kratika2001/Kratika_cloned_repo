from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    def reverse_in_groups(lst, n):
        result = []
        i = 0
        
        # Loop through the list in steps of size n
        while i < len(lst):
            # Reverse the next group of n elements manually
            group = []
            for j in range(min(n, len(lst) - i)):
                group.insert(0, lst[i + j])
            result.extend(group)
            i += n
        
        return result

    return reverse_in_groups(lst, n)

# Test cases
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))  
print(reverse_by_n_elements([1, 2, 3, 4, 5], 2))          
print(reverse_by_n_elements([10, 20, 30, 40, 50, 60, 70], 4)) 



def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    def group_strings_by_length(lst):
    length_dict = {}
    
    # Iterate through the list of strings
    for string in lst:
        length = len(string)
        
        # If the length is not a key in the dictionary, add it
        if length not in length_dict:
            length_dict[length] = []
        
        # Append the string to the appropriate list
        length_dict[length].append(string)
    
    # Sort the dictionary by keys and return it
    return dict(sorted(length_dict.items()))

# Test cases
print(group_strings_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))

print(group_strings_by_length(["one", "two", "three", "four"]))
    return dict

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def flatten_dict(d, parent_key='', sep='.'):
    flattened = {}
    
    # Iterate through each key-value pair in the dictionary
    for k, v in d.items():
        # Create new key for the flattened dictionary
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            # Recursively flatten nested dictionaries
            flattened.update(flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            # Flatten lists by referencing the index
            for i, item in enumerate(v):
                list_key = f"{new_key}[{i}]"
                if isinstance(item, dict):
                    # Recursively flatten if the list contains a dictionary
                    flattened.update(flatten_dict(item, list_key, sep=sep))
                else:
                    # Directly add non-dictionary items to the flattened dictionary
                    flattened[list_key] = item
        else:
            # Add non-dictionary, non-list items to the flattened dictionary
            flattened[new_key] = v
    
    return flattened

# Test case
nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

# Flatten the nested dictionary
flattened_dict = flatten_dict(nested_dict)
print(flattened_dict)


    return dict

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Sort the list to ensure duplicates are adjacent
    nums.sort()
    
    result = []
    used = [False] * len(nums)
    
    def backtrack(path):
        # If the path is the same length as the input, add it to the result
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        # Iterate through the list and build permutations
        for i in range(len(nums)):
            # Skip duplicates or already used elements
            if used[i]:
                continue
 if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue
            
            # Mark this element as used
            used[i] = True
            # Add the current number to the permutation
            path.append(nums[i])
            # Recursively build the rest of the permutation
            backtrack(path)
            # Backtrack: remove the number and mark it as unused
            path.pop()
            used[i] = False
    
    # Start backtracking with an empty path
    backtrack([])
    
    return result

# Test cases
print(unique_permutations([1, 1, 2]))  
print(unique_permutations([1, 2, 3])) 

import re

def find_all_dates(text: str) -> List[str]:
    """
    Find all dates in the text in the formats dd-mm-yyyy, mm/dd/yyyy, yyyy.mm.dd.
    
    :param text: Input string that may contain dates in various formats
    :return: List of valid dates found in the text
    """
    # Define a regex pattern for the three date formats
    date_pattern = r'(\b\d{2}-\d{2}-\d{4}\b)|(\b\d{2}/\d{2}/\d{4}\b)|(\b\d{4}\.\d{2}\.\d{2}\b)'
    
    # Find all matches of the pattern in the text
    matches = re.findall(date_pattern, text)
    
    # Extract the non-empty matches and flatten the result
    dates = [match for group in matches for match in group if match]
    
    return dates

# Test case
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
print(find_all_dates(text))  # Output: ['23-08-1994', '08/23/1994', '1994.08.23']

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    return pd.Dataframe()


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    import polyline
    import pandas as pd
    import math

# Haversine function to calculate the distance between two lat/lon points
def haversine(lat1, lon1, lat2, lon2):
    # Radius of Earth in meters
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c  # Output distance in meters
    return distance
# Function to decode polyline, convert to DataFrame, and calculate distances
def decode_polyline_to_dataframe(polyline_str):
    # Decode polyline into a list of (latitude, longitude) tuples
    coordinates = polyline.decode(polyline_str)
    
    # Convert the list into a Pandas DataFrame
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Add a 'distance' column, initialized with NaN
    df['distance'] = 0.0
    
    # Calculate distances using Haversine formula
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i - 1, 'latitude'], df.loc[i - 1, 'longitude']
        lat2, lon2 = df.loc[i, 'latitude'], df.loc[i, 'longitude']
        df.loc[i, 'distance'] = haversine(lat1, lon1, lat2, lon2)
    
    return df
# Example polyline string
polyline_str = '_p~iF~ps|U_ulLnnqC_mqNvxq`@'

# Decode and convert to DataFrame
df = decode_polyline_to_dataframe(polyline_str)

# Display the DataFrame
print(df)


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    import pandas as pd

def verify_time_coverage(df):
    # Convert the timestamp columns to datetime
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
    # Function to check if the time data covers all 7 days and 24 hours
    def check_full_coverage(sub_df):
        # Set to track covered days and hours
        covered_days = set()
        covered_hours = set()
        
        for _, row in sub_df.iterrows():
            start_time = row['start_datetime']
            end_time = row['end_datetime']
            
            # Loop through the range from start to end time (by hour)
            current_time = start_time
            while current_time <= end_time:
                covered_days.add(current_time.weekday())  # Weekday (0=Monday, 6=Sunday)
                covered_hours.add(current_time.hour)      # Hour (0-23)
                current_time += pd.Timedelta(hours=1)
                import pandas as pd

def verify_time_coverage(df):
    # Convert the timestamp columns to datetime
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
    # Function to check if the time data covers all 7 days and 24 hours
    def check_full_coverage(sub_df):
        # Set to track covered days and hours
        covered_days = set()
        covered_hours = set()
        
        for _, row in sub_df.iterrows():
            start_time = row['start_datetime']
            end_time = row['end_datetime']
            
            # Loop through the range from start to end time (by hour)
            current_time = start_time
            while current_time <= end_time:
                covered_days.add(current_time.weekday())  # Weekday (0=Monday, 6=Sunday)
                covered_hours.add(current_time.hour)      # Hour (0-23)
                current_time += pd.Timedelta(hours=1)
# Check if all 7 days (0-6) and all 24 hours (0-23) are covered
        return len(covered_days) == 7 and len(covered_hours) == 24
    
    # Group by (id, id_2) and apply the coverage check function
    result = df.groupby(['id', 'id_2']).apply(check_full_coverage)
    
    # Return the boolean series indicating if each group has incomplete coverage
    return ~result  

