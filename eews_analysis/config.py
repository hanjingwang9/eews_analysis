# -- Google Cloud Settings --
PROJECT_ID = "analyzing-eews"
LOCATION = "us-central1"
BUCKET = "turkey_tweets_0"

# -- GCS Folder Paths --
INPUT_PATH_1 = "INPUTS_1"
INPUT_PATH_2 = "INPUTS_2"
OUTPUT_PATH = "OUTPUTS_1"
DELETED_FILES_PATH = "DELETED_FILES"
VISUALIZATION_PATH = "VISUALIZATIONS_3"

# -- File Names --
RESULTS_FILENAME = "test_results.json"
BACKUP_FILENAME = "test_results_original_backup.json"

# -- Model & Generation Settings --
MODEL_NAME = "gemini-1.5-flash-001"
GENERATION_CONFIG = {
    "max_output_tokens": 8192,
    "temperature": 0.2,
    "top_p": 0.95,
}