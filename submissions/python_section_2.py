import pandas as pd
import numpy as np
import time


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    unique_ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    distance_matrix = pd.DataFrame(np.inf, index=unique_ids, columns=unique_ids)
    np.fill_diagonal(distance_matrix.values, 0)

    for _, row in df.iterrows():
        distance_matrix.loc[row['id_start'], row['id_end']] = row['distance']
        distance_matrix.loc[row['id_end'], row['id_start']] = row['distance']

    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if distance_matrix.loc[i, k] + distance_matrix.loc[k, j] < distance_matrix.loc[i, j]:
                    distance_matrix.loc[i, j] = distance_matrix.loc[i, k] + distance_matrix.loc[k, j]

    return distance_matrix



def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here

    unrolled_df = df.reset_index().melt(id_vars='index',
                                                     var_name='id_end',
                                                     value_name='distance')

    unrolled_df = unrolled_df.rename(columns={'index': 'id_start'})

    unrolled_df = unrolled_df[unrolled_df['id_start'] != unrolled_df['id_end']]
    unrolled_df = unrolled_df[unrolled_df['distance'] != float('inf')]

    return unrolled_df

# distance_matrix = calculate_distance_matrix(df)
# unroll_distance_matrix(distance_matrix)


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here

    ref_distances = df[df['id_start'] == reference_id]

    # Calculate the average distance for the reference_id
    if ref_distances.empty:
        return []  # Return empty if no distances found for the reference ID

    average_distance = ref_distances['distance'].mean()

    # Calculate the threshold values (10% above and below the average distance)
    lower_bound = average_distance * 0.9
    upper_bound = average_distance * 1.1

    # Find IDs within the threshold
    within_threshold = df[
        (df['id_start'] != reference_id) &  # Exclude the reference ID itself
        (df['distance'] >= lower_bound) &
        (df['distance'] <= upper_bound)
        ]

    # Get the unique id_start values within the threshold and sort them
    result_ids = sorted(within_threshold['id_start'].unique())

    return result_ids


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here

    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates for each vehicle type
    for vehicle, coefficient in rate_coefficients.items():
        df[vehicle] = df['distance'] * coefficient

    return df.drop(['distance'], axis=1)

# unrolled_df = unroll_distance_matrix(df)
# toll_rates_df = calculate_toll_rate(unrolled_df)
# print(toll_rates_df)


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here

    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Define time intervals and discount factors
    time_intervals = [
        (time(0, 0), time(10, 0), 0.8),  # Weekdays morning discount
        (time(10, 0), time(18, 0), 1.2),  # Weekdays day rate
        (time(18, 0), time(23, 59, 59), 0.8)  # Weekdays evening discount
    ]
    weekend_discount = 0.7  # Constant for weekends

    # Prepare a list to hold the new rows
    new_rows = []

    # Iterate through each unique id_start and id_end pair
    for (id_start, id_end), group in df.groupby(['id_start', 'id_end']):
        for day in days_of_week:
            for start_time, end_time, factor in time_intervals:
                # Add weekday entries
                new_rows.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                    'moto': group['moto'].values[0] * factor,
                    'car': group['car'].values[0] * factor,
                    'rv': group['rv'].values[0] * factor,
                    'bus': group['bus'].values[0] * factor,
                    'truck': group['truck'].values[0] * factor
                })

        # Add weekend entries
        for day in ['Saturday', 'Sunday']:
            new_rows.append({
                'id_start': id_start,
                'id_end': id_end,
                'start_day': day,
                'start_time': time(0, 0),
                'end_day': day,
                'end_time': time(23, 59, 59),
                'moto': group['moto'].values[0] * weekend_discount,
                'car': group['car'].values[0] * weekend_discount,
                'rv': group['rv'].values[0] * weekend_discount,
                'bus': group['bus'].values[0] * weekend_discount,
                'truck': group['truck'].values[0] * weekend_discount
            })

    # Create a new DataFrame from the new rows
    time_based_toll_df = pd.DataFrame(new_rows)

    return time_based_toll_df

# unrolled_df = unroll_distance_matrix(df)
# toll_rates_time_based_df = calculate_toll_rate(unrolled_df)
# toll_rates_time_based_df = calculate_time_based_toll_rates(toll_rates_df)
# print(toll_rates_time_based_df)