import json
import os
from google.cloud import storage
import pandas as pd
from datetime import datetime
import numpy as np
from eews_analysis import config

def clean_data():
    try:
        storage_client = storage.Client(project=config.PROJECT_ID)
        bucket = storage_client.bucket(config.BUCKET)
        print("GCS Client initialized successfully.")
    except Exception as e:
        print(f"Could not initialize GCS Client. Error: {e}")
        raise

    source_path = os.path.join(config.OUTPUT_PATH, config.RESULTS_FILENAME)

    print(f"\n--- Step 2: Loading data from {source_path} ---")
    try:
        source_blob = bucket.blob(source_path)
        if not source_blob.exists():
            raise FileNotFoundError(f"Source file not found at gs://{config.bucket}/{source_path}")

        json_data_string = source_blob.download_as_string()
        all_data = json.loads(json_data_string)
        df = pd.DataFrame.from_dict(all_data, orient='index')
        print(f"Successfully loaded data for {len(df)} entries.")
    except Exception as e:
        print(f"Could not load JSON data. Reason: {e}.")
        df = None

    if df is not None:
        print("\n--- Step 3: Filtering data based on magnitude and date criteria ---")
        initial_rows = len(df)

        df['mag_numeric'] = pd.to_numeric(df['magnitude_on_alert_screenshot'], errors='coerce')
        df['post_datetime_dt'] = pd.to_datetime(df['post_datetime'], errors='coerce')
        df['alert_time_dt'] = pd.to_datetime(df['alert_time'], errors='coerce')

        magnitude_remove_mask = ((df['mag_numeric'] > 6.2) | (df['mag_numeric'] <= 4.5))

        start_bound = datetime(2025, 4, 1)
        end_bound = datetime(2025, 6, 1)
        post_date_remove_mask = ((df['post_datetime_dt'] < start_bound) | (df['post_datetime_dt'] >= end_bound))
        alert_date_remove_mask = ((df['alert_time_dt'] < start_bound) | (df['alert_time_dt'] >= end_bound))
        date_remove_mask = post_date_remove_mask | alert_date_remove_mask

        combined_remove_mask = magnitude_remove_mask | date_remove_mask

        df_to_remove = df[combined_remove_mask]
        df_to_keep = df[~combined_remove_mask]

        filenames_to_remove = df_to_remove.index.tolist()

        print(f"Filtering complete.")
        print(f"   - Entries to keep: {len(df_to_keep)}")
        print(f"   - Entries to remove: {len(df_to_remove)}")

    if df is not None and filenames_to_remove:
        print(f"\n--- Step 4: Moving {len(filenames_to_remove)} PNG files to gs://{config.OUTPUTS_PATH}/{config.DELETED_FILES_PATH}/ ---")

        moved_count = 0
        for filename in filenames_to_remove:
            source_path_1 = os.path.join(config.INPUT_PATH_1, filename)
            source_path_2 = os.path.join(config.INPUT_PATH_2, filename)

            source_blob = None
            source_blob_gcs = bucket.blob(source_path_1)
            if source_blob_gcs.exists():
                source_blob = source_blob_gcs
            else:
                source_blob_gcs = bucket.blob(source_path_2)
                if source_blob_gcs.exists():
                    source_blob = source_blob_gcs

            if source_blob:
                destination_path = os.path.join(config.DELETED_FILES_PATH, filename)
                bucket.copy_blob(source_blob, bucket, destination_path)
                source_blob.delete()
                moved_count += 1
            else:
                print(f"   - Warning: Could not find source PNG file for '{filename}' in either input folder.")

        print(f"Successfully moved {moved_count} PNG files.")

    if df is not None:
        print(f"\n--- Step 5: Saving data ---")

        original_blob = bucket.blob(source_path)
        if original_blob.exists():
            backup_path = os.path.join(config.OUTPUT_PATH, config.BACKUP_FILENAME)
            bucket.rename_blob(original_blob, new_name=backup_path)
            print(f"Original data archived to: gs://{config.BUCKET}/{backup_path}")

        df_to_keep = df_to_keep.drop(columns=['mag_numeric', 'post_datetime_dt', 'alert_time_dt'])
        cleaned_data_dict = df_to_keep.to_dict(orient='index')

        json_string = json.dumps(cleaned_data_dict, indent=2)
        bucket.blob(source_path).upload_from_string(json_string, content_type="application/json")

        print(f"Cleaned data for {len(df_to_keep)} entries saved to: gs://{config.BUCKET}/{source_path}")


if __name__ == "__main__":
    clean_data()