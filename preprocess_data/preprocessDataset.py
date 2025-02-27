import pandas as pd
import glob
import os


def addLabel(df, user_number):
    df_labels = pd.read_csv(f"data/raw/user-{user_number}/activity_labels.csv")
    df_data = df

    # Initialize label column
    df_data["label"] = "UNKNOWN"  # Default label if no match is found

    # Assign labels based on the timestamp interval
    for _, row in df_labels.iterrows():
        mask = (df_data["ts"] >= row["ts_start"]) & (df_data["ts"] <= row["ts_end"])
        df_data.loc[mask, "label"] = row["label"]

    # Save the labeled dataset
    df_data.to_csv(f"data/labeled_sensor_data_merged_user-{user_number}.csv", index=False)

    return df_data



def load_dataset_by_id(user_id):
    # Get all CSV files that start with "smart"
    file_list = glob.glob(f"data/raw/user-{user_id}/smart*.csv")
    print(f"reading csv for user: {user_id}")
    files = {f: pd.read_csv(f) for f in file_list}

    return files


def identify_and_rename_datasets(files):

    # Identify datasets by checking filenames
    df_p_acc = next((df for fname, df in files.items() if "phone_acc" in fname.lower()), None)
    df_p_gyro = next((df for fname, df in files.items() if "phone_gyr" in fname.lower()), None)
    df_p_mag = next((df for fname, df in files.items() if "phone_mag" in fname.lower()), None)

    df_w_acc = next((df for fname, df in files.items() if "watch_acc" in fname.lower()), None)
    df_w_gyro = next((df for fname, df in files.items() if "watch_gyr" in fname.lower()), None)
    df_w_mag = next((df for fname, df in files.items() if "watch_mag" in fname.lower()), None)

    # Ensure all datasets are sorted by timestamp
    for df in [df_p_acc, df_p_gyro, df_p_mag, df_w_acc, df_w_gyro, df_w_mag]:
        if df is not None:
            df.sort_values("ts", inplace=True)

    # Rename columns for clarity before merging
    df_p_acc.rename(columns={'x': 'p_acc_x', 'y': 'p_acc_y', 'z': 'p_acc_z'}, inplace=True)
    df_p_gyro.rename(columns={'x': 'p_gyro_x', 'y': 'p_gyro_y', 'z': 'p_gyro_z'}, inplace=True)
    df_p_mag.rename(columns={'x': 'p_mag_x', 'y': 'p_mag_y', 'z': 'p_mag_z'}, inplace=True)

    df_w_acc.rename(columns={'x': 'w_acc_x', 'y': 'w_acc_y', 'z': 'w_acc_z'}, inplace=True)
    df_w_gyro.rename(columns={'x': 'w_gyro_x', 'y': 'w_gyro_y', 'z': 'w_gyro_z'}, inplace=True)
    df_w_mag.rename(columns={'x': 'w_mag_x', 'y': 'w_mag_y', 'z': 'w_mag_z'}, inplace=True)

    return [df_p_acc, df_p_gyro, df_p_mag, df_w_acc, df_w_gyro, df_w_mag]


def merge_nearest_ts(dfs):
    # Merge datasets on nearest timestamp
    df_merged = dfs[0].copy()
    for df in dfs[1:]:
        if df is not None:
            df_merged = pd.merge_asof(df_merged, df, on="ts", direction="nearest")

    # Save the merged dataset
    # df_merged.to_csv("merged_sensors.csv", index=False)

    return df_merged


def get_num_users():
    return len([d for d in os.listdir("data/raw") if os.path.isdir(os.path.join("data/raw", d))])



if __name__ == '__main__':
    num_users = get_num_users()
    for user_id in range(1, num_users + 1):
        files = load_dataset_by_id(user_id)
        datasets = identify_and_rename_datasets(files)
        merged_df = merge_nearest_ts(datasets)
        addLabel(merged_df, f"{user_id}")
