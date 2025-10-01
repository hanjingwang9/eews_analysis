import json
import os
from google.cloud import storage
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np

from eews_analysis import config

def load_and_preprocess_data(bucket):
    try:
        blob = bucket.blob(os.path.join(config.OUTPUT_PATH, config.RESULTS_FILENAME))
        json_data_string = blob.download_as_string()
        all_data = json.loads(json_data_string)
        df = pd.DataFrame.from_dict(all_data, orient='index')
        df['post_datetime'] = pd.to_datetime(df['post_datetime'], errors='coerce')
        print(f"Successfully loaded data for {len(df)} entries.")
    except Exception as e:
        print(f"Could not load JSON data. Reason: {e}.")
        return None

    df = df[df['alert_source'].isin(['AEA', 'UNKNOWN', 'NOT_APPLICABLE', None])]
    df.loc[df["user's_past_earthquake_experience"] == 'YES', "user's_past_earthquake_experience"] = 'UNKNOWN'

    location_replacements = {
        'Near Antalya': 'Antalya', 'Mamak': 'Ankara', 'Antalya, Mugla, Izmir': 'Izmir',
        'Izmir karsiyaka': 'Izmir', 'Izmir guzelbache': 'Izmir', 'west of Istanbul': 'Istanbul',
        'Near Istanbul': 'Istanbul', 'Istanbul, Turkieh': 'Istanbul', 'Silivri, Istanbul': 'Silivri',
        'Silivri, Marmara': 'Silivri', 'near Silivri': 'Silivri', 'mugla, turkey': 'Mugla',
        'mugla fethiye': 'Mugla', 'eastern mediterranean sea': 'eastern mediterranean',
        'Near the coast of Turkey': 'Edge of the marked area', 'Turkey': 'Edge of the marked area',
    }
    df['user_approximate_location_on_alert'] = df['user_approximate_location_on_alert'].replace(location_replacements)
    df = df[df['user_approximate_location_on_alert'] != 'Edge of the marked area']

    df['location_combined'] = df['user_approximate_location_on_alert'].replace({'UNKNOWN': np.nan, 'NOT_APPLICABLE': np.nan})
    df['location_combined'] = df['location_combined'].fillna(df['post_location'])

    numeric_warning_times = pd.to_numeric(df['warning_time_seconds'], errors='coerce')
    bins = [-np.inf, 5, 15, 30, 60, 120, np.inf]
    labels = ["<5 sec", "5-15 sec", "15-30 sec", "30-60 sec", "60-120 sec", ">120 sec"]
    df['warning_time_binned'] = pd.cut(numeric_warning_times, bins=bins, labels=labels, right=False)
    
    print("Preprocessing complete.")
    return df

def save_and_upload(fig, title, bucket, file_type='html'):
    local_filename = f"{title.replace(' ', '_').replace(':', '').lower()}.{file_type}"
    gcs_path = os.path.join(config.VISUALIZATION_PATH, local_filename)

    if file_type == 'html':
        fig.write_html(local_filename)
        content_type = 'text/html'
    elif file_type == 'png':
        fig.write_image(local_filename, scale=2, width=1000, height=1200)
        content_type = 'image/png'
    else:
        print(f"Unsupported file type: {file_type}")
        return

    bucket.blob(gcs_path).upload_from_filename(local_filename, content_type=content_type)
    print(f"Chart '{title}' uploaded to gs://{config.BUCKET}/{gcs_path}")

    if os.path.exists(local_filename):
        os.remove(local_filename)

def create_sunburst(data_df, path, title, bucket, text_info=None):
    font_settings = dict(family="Arial, sans-serif", size=14, color="black")
    plot_df = data_df[path].copy()
    
    for col in path:
        plot_df[col] = plot_df[col].astype(str)
        plot_df = plot_df[~plot_df[col].isin(["UNKNOWN", "NOT_APPLICABLE", 'nan', 'None'])]
    
    plot_df.dropna(inplace=True)
    if len(plot_df) < 20:
        print(f"Skipping Sunburst for '{title}': Not enough valid data.")
        return

    fig = px.sunburst(plot_df, path=path, color=path[0])
    sample_count = len(plot_df)
    formatted_title = f"{title} (n={sample_count})"
    
    all_parents = set(fig.data[0].parents)
    custom_text_templates = ['%{label}<br>%{percentRoot:.1%}' if i not in all_parents else '%{label}' for i in fig.data[0].ids]
    fig.update_traces(texttemplate=custom_text_templates, textfont_size=12, insidetextorientation='radial')

    fig.update_layout(
        title_text=formatted_title, font=font_settings, margin=dict(t=60, l=10, r=10, b=10),
        annotations=[dict(text=f'n={sample_count}', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    if text_info:
        fig.update_traces(textinfo=text_info)

    save_and_upload(fig, title, bucket, 'html')

def plot_sample_sizes(df, bucket):
    sample_sizes = {}
    for col in df.columns:
        if col not in ["username", "post_datetime", "reasoning"]:
            valid_data = df[col].dropna()
            valid_data = valid_data[~valid_data.isin(["UNKNOWN", "NOT_APPLICABLE", None])]
            if valid_data.apply(isinstance, args=(list,)).any():
                valid_data = valid_data.explode()
            sample_sizes[col.replace('_', ' ').title()] = len(valid_data)
    
    sample_df = pd.DataFrame(list(sample_sizes.items()), columns=['Attribute', 'Sample Size (n)']).sort_values('Sample Size (n)', ascending=True)
    fig = px.bar(sample_df, x='Sample Size (n)', y='Attribute', orientation='h', title='Available Samples per Question', text='Sample Size (n)')
    fig.update_layout(font_size=14, yaxis={'categoryorder':'total ascending'})
    fig.update_traces(textposition='inside', marker_color='skyblue')
    save_and_upload(fig, "sample_size_overview", bucket, 'png')

def plot_general_visualizations(df, bucket):
    excluded = {"username", "alert_language", "reasoning", "magnitude_on_alert_screenshot", "warning_time_seconds", "distance_on_alert_screenshot_ml"}
    numerical = {"shaking_intensity_mmi"}
    font_settings = dict(family="Arial, sans-serif", size=16, color="black")

    for attribute in [col for col in df.columns if col not in excluded]:
        valid_data = df[attribute].dropna()
        valid_data = valid_data[~valid_data.isin(["UNKNOWN", "NOT_APPLICABLE", None])]
        if valid_data.empty: continue

        sample_count = len(valid_data)
        title = f"{attribute.replace('_', ' ').title()} (n={sample_count})"
        fig = None

        try:
            if attribute in numerical:
                plot_data = pd.to_numeric(valid_data, errors='coerce').dropna()
                if plot_data.empty: continue
                fig = px.histogram(plot_data, nbins=20, title=f"{title} (n={len(plot_data)})")
                fig.update_layout(bargap=0.1, yaxis_title="Frequency", xaxis_title=attribute.replace('_', ' ').title(), showlegend=False)
            else:
                if valid_data.apply(isinstance, args=(list,)).any():
                    valid_data = valid_data.explode()
                if attribute == 'alert_info_recall':
                    valid_data = valid_data[valid_data != 'UNKNOWN']
                
                sample_count = len(valid_data)
                value_counts_df = valid_data.value_counts().reset_index().head(10)
                value_counts_df.columns = ['category', 'count']
                fig = px.pie(value_counts_df, values='count', names='category', title=title, hole=0.4)
                fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14)
                fig.update_layout(annotations=[dict(text=f'n={sample_count}', x=0.5, y=0.5, font_size=24, showarrow=False)], legend_title_text='Categories')
            
            if fig:
                fig.update_layout(font=font_settings)
                save_and_upload(fig, attribute, bucket, 'png')
        except Exception as e:
            print(f"Could not create plot for '{attribute}'. Reason: {e}")

def plot_event_specific(df, bucket):
    date_ranges = {"april_23_to_24": (datetime(2025, 4, 23), datetime(2025, 4, 24, 23, 59, 59))}
    hist_attrs = ["magnitude_on_alert_screenshot", "warning_time_seconds", "distance_on_alert_screenshot_ml"]
    sunburst_rels = {
        "Location and Alert Type": ["location_combined", "alert_type"], "Location and User Sentiment": ["location_combined", "users_sentiment"],
        "Shaking Level and Warning Time": ["shaking_level", "warning_time_binned"], "Location and Warning Time": ["location_combined", "warning_time_binned"]
    }
    font_settings = dict(family="Arial, sans-serif", size=16, color="black")

    for name, (start, end) in date_ranges.items():
        event_df = df.dropna(subset=['post_datetime'])
        event_df = event_df[(event_df['post_datetime'] >= start) & (event_df['post_datetime'] <= end)]
        if event_df.empty: continue

        title_prefix = f"{name.replace('_', ' ').title()}"
        for attr in hist_attrs:
            valid_data = pd.to_numeric(event_df[attr], errors='coerce').dropna()
            if valid_data.empty: continue
            title = f"{title_prefix}: {attr.replace('_', ' ').title()}"
            fig = px.histogram(valid_data, nbins=20, title=f"{title} (n={len(valid_data)})")
            fig.update_layout(font=font_settings, bargap=0.1, yaxis_title="Frequency", xaxis_title=attr.replace('_', ' ').title(), showlegend=False)
            save_and_upload(fig, f"{name}_{attr}", bucket, 'png')
        
        for title, path in sunburst_rels.items():
            create_sunburst(event_df, path, f"{title_prefix}: {title}", bucket)

def plot_nested_relationships(df, bucket):
    relationships = {
        "Shaking Level and Alert Arrival": ["shaking_level", "alert_arrival_wrt_shaking"], "User Sentiment and Gender": ["users_sentiment", "user_s_gender"],
        "Action Taken and Gender": ["post_alert_action", "user_s_gender"], "Alert Type and Mode": ["alert_type", "alert_mode"],
        "Screenshot with Contour": ["with_alert_screenshot", "alert_screenshot_with_contour"], "Gender and Helpfulness": ["user's_gender", "helpfulness"],
        "Gender and Alert Info Recall": ["user's_gender", "alert_info_recall"], "Gender and Post Alert Action": ["user's_gender", "post_alert_action"],
        "Location and Alert Type": ["location_combined", "alert_type"], "Location and User Sentiment": ["location_combined", "users_sentiment"],
        "Shaking Level and Warning Time": ["shaking_level", "warning_time_binned"], "Location and Warning Time": ["location_combined", "warning_time_binned"],
        "Shaking Experience and MMI": ["felt_shaking", "shaking_level", "shaking_intensity_mmi"], "Location and MMI": ["location_combined", "shaking_intensity_mmi"],
        "Location and Alert Arrival": ["location_combined", "alert_arrival_wrt_shaking"]
    }
    for title, path in relationships.items():
        if all(col in df.columns for col in path):
            create_sunburst(df, path, title, bucket)
    
    create_sunburst(df, ["helpfulness", "helpfulness_reason"], "Helpfulness and Reason", bucket, text_info="label+percent parent")

def perform_sanity_checks(df):
    print("\n--- Data Sanity Checks ---")
    df['magnitude_on_alert_screenshot'] = pd.to_numeric(df['magnitude_on_alert_screenshot'], errors='coerce')
    mag_check = df.dropna(subset=['magnitude_on_alert_screenshot'])
    mag_check = mag_check[(mag_check['magnitude_on_alert_screenshot'] <= 5) | (mag_check['magnitude_on_alert_screenshot'] >= 6.2)]
    if not mag_check.empty:
        print("\n--- Checking Magnitude Data (<=5 or >=6.2) ---")
        print(mag_check[['post_datetime', 'magnitude_on_alert_screenshot', 'reasoning']])
    
    strong_before = df[(df['shaking_level'] == 'STRONG') & (df['alert_arrival_wrt_shaking'] == 'BEFORE_SHAKING')]
    if not strong_before.empty:
        print("\n--- Checking 'Strong - Before Shaking' Data ---")
        for index, row in strong_before.iterrows():
            print(f"Filename: {index}\\nReasoning: {row['reasoning']}\\n---")

def main():
    try:
        storage_client = storage.Client(project=config.PROJECT_ID)
        bucket = storage_client.bucket(config.BUCKET)
    except Exception as e:
        print(f"Failed to initialize GCS client. Error: {e}")
        return

    df = load_and_preprocess_data(bucket)

    if df is not None:
        plot_sample_sizes(df, bucket)
        plot_general_visualizations(df, bucket)
        plot_event_specific(df, bucket)
        plot_nested_relationships(df, bucket)
        perform_sanity_checks(df)

if __name__ == "__main__":
    main()