from typing import Dict, List, Set, Tuple
import polyline
import pandas as pd
import numpy as np


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    lst[:] = list(reversed(lst[:n])) + lst[n:]
    return lst


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    len_dict = {}
    for word in lst:
        length = len(word)
        if length not in len_dict:
            len_dict[length] = []
        len_dict[length].append(word)
    return dict(sorted(len_dict.items()))


def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    flattened = {}
    stack = [(nested_dict, '')]  # Stack to hold dictionaries and their parent keys

    while stack:
        current_dict, parent_key = stack.pop()

        for key, value in current_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, dict):
                stack.append((value, new_key))  # Push the nested dictionary onto the stack
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        stack.append((item, f"{new_key}[{i}]"))  # Handle dictionaries in lists
                    else:
                        flattened[f"{new_key}[{i}]"] = item  # Handle non-dict items in lists
            else:
                flattened[new_key] = value  # Add the flattened key-value pair

    return flattened


def unique_permutations(nums: List[int]) -> set[tuple[int, ...]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    import itertools
    return set(itertools.permutations(nums))


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    import re
    patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b'  # yyyy.mm.dd
    ]

    # Combine all patterns into a single pattern
    combined_pattern = '|'.join(patterns)

    # Find all matches in the input text
    matches = re.findall(combined_pattern, text)

    return matches


def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    decoded_points = polyline.decode(polyline_str)

    if not decoded_points:
        raise ValueError("Decoded polyline contains no points.")

    df = pd.DataFrame(decoded_points, columns=['lat', 'long'])

    distances = [0.0]  # First distance is 0

    for i in range(1, len(df)):
        lat1, lon1 = df.iloc[i - 1][['lat', 'long']]
        lat2, lon2 = df.iloc[i][['lat', 'long']]

        # Haversine calculation
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371000  # Earth radius in meters
        distance = c * r

        distances.append(distance)

    df['distance'] = distances

    return df[['lat', 'long', 'distance']]


# print(polyline_to_dataframe('onl~Fj|cvOrsEg}@rHuvK'))


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    n = len(matrix)

    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

    # Step 2: Transform the rotated matrix
    final_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            # Calculate sum of the row and column excluding the current element
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum

    return final_matrix


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    index = pd.MultiIndex.from_frame(df[['id', 'id_2']].drop_duplicates())

    results = pd.Series(False, index=index)

    for (id_val, id_2_val), group in df.groupby(['id', 'id_2']):
        # Initialize sets for days and time coverage
        days_covered = set()
        time_ranges = []

        for _, row in group.iterrows():
            days_covered.add(row['startDay'])
            days_covered.add(row['endDay'])

            time_ranges.append((row['startTime'], row['endTime'], row['startDay'], row['endDay']))

        all_days = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'}
        is_days_covered = days_covered >= all_days

        is_full_24_hours = False

        # Normalize time strings to seconds for comparison
        time_slots = []
        for start_time, end_time, start_day, end_day in time_ranges:
            start_seconds = int(start_time.split(':')[0]) * 3600 + int(start_time.split(':')[1]) * 60 + int(
                start_time.split(':')[2])
            end_seconds = int(end_time.split(':')[0]) * 3600 + int(end_time.split(':')[1]) * 60 + int(
                end_time.split(':')[2])
            time_slots.append((start_seconds, end_seconds))

        total_time_covered = [0] * 86400

        for start_seconds, end_seconds in time_slots:
            if start_seconds < end_seconds:
                for sec in range(start_seconds, end_seconds):
                    total_time_covered[sec] = 1
            else:  # Handle overnight spans (e.g., 23:00 to 01:00)
                for sec in range(start_seconds, 86400):
                    total_time_covered[sec] = 1
                for sec in range(0, end_seconds):
                    total_time_covered[sec] = 1

        is_full_24_hours = all(total_time_covered)
        results[(id_val, id_2_val)] = not (is_days_covered and is_full_24_hours)
    return results
