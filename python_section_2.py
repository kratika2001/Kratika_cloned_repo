import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    def calculate_distance_matrix(df):
    # Extract unique IDs from the dataset
    ids = pd.unique(df[['id_from', 'id_to']].values.ravel('K'))
    
    # Create an empty distance matrix with 0.0 for the diagonal
    distance_matrix = pd.DataFrame(0.0, index=ids, columns=ids)
    
    # Fill in the known distances
    for _, row in df.iterrows():
        distance_matrix.loc[row['id_from'], row['id_to']] = row['distance']
        distance_matrix.loc[row['id_to'], row['id_from']] = row['distance']  # Ensure symmetry
    
    # Calculate cumulative distances
    for k in ids:
        for i in ids:
 for j in ids:
                # If there is a path from i to k and k to j, update the distance
                if distance_matrix.loc[i, k] > 0 and distance_matrix.loc[k, j] > 0:
                    distance_matrix.loc[i, j] = min(distance_matrix.loc[i, j], distance_matrix.loc[i, k] + distance_matrix.loc[k, j])
    
    return distance_matrix


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    def unroll_distance_matrix(distance_matrix):
    # Create a list to hold the results
    results = []
    
    # Iterate through the distance matrix
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:  # Exclude the same ID comparisons
                distance = distance_matrix.loc[id_start, id_end]
                results.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})
    
    # Create a DataFrame from the results
    unrolled_df = pd.DataFrame(results)
    
    return unrolled_df


def find_ids_within_ten_percentage_threshold(ids, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    def find_ids_within_ten_percentage_threshold(unrolled_ids, reference_id):
    # Filter rows where id_start is the reference_id
    distances = unrolled_df[unrolled_ids['id_start'] == reference_id]['distance']
    
    # Check if there are any distances for the given reference_id
    if distances.empty:
        return []  # Return an empty list if no distances found for the reference_id
    
    # Calculate the average distance for the reference_id
    average_distance = distances.mean()
    
    # Calculate the lower and upper bounds (10% threshold)
    lower_bound = average_distance * 0.9
    upper_bound = average_distance * 1.1
    
    # Find all id_start values that fall within the threshold
    filtered_ids = unrolled_ids[(unrolled_ids['distance'] >= lower_bound) &
                                (unrolled_ids['distance'] <= upper_bound)]
    
    # Get the unique id_start values and sort them
    result_ids = sorted(filtered_ids['id_start'].unique())
    
    return result_ids


def calculate_toll_rate( unrolled_df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    def calculate_toll_rate(unrolled_df):
    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Calculate toll rates for each vehicle type
    for vehicle, coefficient in rate_coefficients.items():
        unrolled_df[vehicle] = unrolled_df['distance'] * coefficient
    
    return unrolled_df


def calculate_time_based_toll_rates(result_df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    def calculate_time_based_toll_rates(toll_rates_df):
    # Define discount factors for weekdays
    weekday_discount = {
        (datetime.time(0, 0), datetime.time(10, 0)): 0.8,
        (datetime.time(10, 0), datetime.time(18, 0)): 1.2,
        (datetime.time(18, 0), datetime.time(23, 59, 59)): 0.8
    }
    
    # Constant discount factor for weekends
    weekend_discount = 0.7
    
    # Days of the week
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Prepare a list to hold new rows
    new_rows = []
    
    for index, row in toll_rates_df.iterrows():
        # Extract the id_start and id_end
        id_start = row['id_start']
        id_end = row['id_end']
distance = row['distance']
        
        # Loop over each day of the week
        for day in days:
            # Determine if the day is a weekday or weekend
            if day in days[0:5]:  # Weekdays
                for time_range, discount in weekday_discount.items():
                    start_time, end_time = time_range
                    # Calculate discounted toll rates
                    new_row = {
                        'id_start': id_start,
                        'id_end': id_end,
                        'distance': distance,
                        'start_day': day,
                        'start_time': start_time,
                        'end_day': day,
                        'end_time': end_time,
                        'moto': distance * 0.8 * discount,  # Applying discount
                        'car': distance * 1.2 * discount,
                        'rv': distance * 1.5 * discount,
                        'bus': distance * 2.2 * discount,
                        'truck': distance * 3.6 * discount
                    }
new_rows.append(new_row)
            else:  # Weekends
                start_time = datetime.time(0, 0)
                end_time = datetime.time(23, 59, 59)
                new_row = {
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': distance,
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                    'moto': distance * 0.8 * weekend_discount,
                    'car': distance * 1.2 * weekend_discount,
                    'rv': distance * 1.5 * weekend_discount,
                    'bus': distance * 2.2 * weekend_discount,
                    'truck': distance * 3.6 * weekend_discount
                }
                new_rows.append(new_row)
    
    # Create a new DataFrame with the new rows
    result_df = pd.DataFrame(new_rows)
    
    return result_df
