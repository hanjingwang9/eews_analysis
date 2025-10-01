This project analyzes social media posts (from Twitter/X) to understand public perception and experience with Google's Android Earthquake Alerts (AEA) system in Turkey. It uses Google's Gemini model to extract structured data from tweet screenshots and generates a series of visualizations to explore the findings.

## Features

-   **Data Extraction**: Leverages the Gemini 1.5 Flash model via Vertex AI to parse images of tweets and extract over 40 distinct data points per tweet.
-   **Cloud Integration**: Utilizes Google Cloud Storage (GCS) for storing raw images, processed data (JSON), and final visualizations.
-   **Automated Workflow**: Scripts are provided to automate the entire pipeline from data extraction to visualization.
-   **Data Cleaning**: Includes a script to filter data based on specific criteria (e.g., earthquake magnitude, date) and archive irrelevant source images.
-   **Rich Visualizations**: Generates interactive charts with Plotly, including pie charts, histograms, and multi-level sunburst charts to explore relationships in the data.


## Setup and Installation

### 1. Prerequisites

-   A Google Cloud Platform (GCP) project with the Vertex AI API enabled.
-   A Google Cloud Storage (GCS) bucket.
-   Python 3.8+

### 2. Clone the Repository

```bash
git clone [https://github.com/your-username/eews-social-analysis.git](https://github.com/your-username/eews-social-analysis.git)
cd eews-social-analysis
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Google Cloud Authentication

Authenticate your local environment to access your GCP project.

```bash
gcloud auth application-default login
```

## Configuration

1.  **Set up GCS Bucket**:
    * Create a bucket in your GCP project.
    * Inside the bucket, create the following folders: `INPUTS_1`, `OUTPUTS_1`, `DELETED_FILES`, `VISUALIZATIONS_3`, and `EXAMPLES`.
    * Upload your tweet screenshots to the `INPUTS_1` (and `INPUTS_2` if used) folder.
    * Upload your two few-shot example images to the `EXAMPLES` folder.

2.  **Edit `config.py`**:
    * Update `PROJECT_ID` with your GCP Project ID.
    * Update `BUCKET_NAME` with the name of your GCS bucket.
    * Adjust other path or model settings if necessary.

## Usage

### Step 1: Process Tweets and Extract Data

This script iterates through all images in your `INPUTS` folders, sends them to the Gemini API, and saves the structured JSON output to GCS. The script will automatically skip any images that have already been processed.

```bash
python eews_analysis/process_tweets.py
```

### Step 2: Clean the Extracted Data

This script filters the generated `test_results.json` file based on earthquake magnitude and date ranges defined in the script. It archives the original JSON and moves the corresponding source images of the filtered-out entries.

```bash
python eews_analysis/clean_data.py
```

### Step 3: Generate Visualizations

This script reads the cleaned JSON data and generates a series of interactive HTML charts and PNG images, saving them to the `VISUALIZATIONS_3` folder in your GCS bucket.

```bash
python eews_analysis/visualize.py
```
After running, you can browse the generated files directly in your GCS bucket.
