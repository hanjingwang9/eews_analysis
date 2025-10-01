

import os
import vertexai
from vertexai.generative_models import GenerativeModel, Part, HarmCategory, HarmBlockThreshold
from google.cloud import storage
import time
import json
import itertools
from eews_analysis import config

def process_all_tweets():
    vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)
    storage_client = storage.Client(project=config.PROJECT_ID)
    bucket = storage_client.bucket(config.BUCKET)

    model = GenerativeModel("gemini-2.0-flash-001")




    # Prompt

    prompt_instruction = """
    You are a computational social scientist analyzing the user's experiences about
    the performance of Google’s Earthquake Early Warning system (a.k.a. Android Earthquake
    Alert or AEA) during recent earthquakes in Turkey. Phones plugged in and stationary
    report their availability for earthquake monitoring by AEA. An on-phone detection
    algorithm analyzes acceleration time series for sudden changes indicative of seismic
    P- or S-wave arrivals. Upon detecting a potential event, de-identified parameter
    data is sent to the backend server. This detection capability is deployed as part
    of Google Play Services, core system software, meaning it is on by default for the
    vast majority of Android smartphones and does not require activation or installation
    of any additional application. The servers then match the pattern of phone triggering
    with possible seismic sources in the time-space domain. An earthquake is declared and
    its source parameters (e.g., magnitude, hypocenter, and origin time) are estimated.
    Upon detection of an earthquake, the intensity of the ground shaking and its potential
    extent are estimated. For events with estimated magnitude exceeding M4.5, AEA sends
    two distinct types of alerts to users that are within the impacted area. These include
    “Take Action” and “Be Aware” alerts for users within the regions expected to experience
    moderate or greater (i.e., >= MMI 5) and weak (i.e., MMI 3 or 4), respectively. The
    “Take Action” alert takes over the entire screen of the phone, breaking through any
    do-not-disturb settings and makes a characteristic sound designed to be attention
    grabbing. The “Be Aware” alert appears as a notification similar to other phone or app
    notifications, but with a characteristic sound. Once the shaking has passed, or if the
    alert arrives after shaking, the alerts are replaced by the "Earthquake Occurred"
    notification. The delivered alerts to the Android phone users contain a short summary
    of the event attributes, precautionary instructions, earthquake safety info, and a short
    user survey feedback of the alert delivery. Please use the provided examples as reference
    and extract these information from the input tweet if available: Username; Post’s date-time
    in YYYY-MM-DDTHH:MM format, keeping in mind that screenshots of tweets are taken in EST
    while local time would be in Istanbul time; Geolocation (like the city or town if available);
    How many seconds before the earthquake did they receive the alert?; Does the post include a
    screenshot or picture of the received alert (YES, NO)?; What is the time of the issued alert --
    in YYYY-MM-DDTHH:MM format -- shown on the attached image of the received alert to the post?;
    What is the magnitude of the earthquake shown on the attached image of the received alert to
    the post?; What is the distance (in miles) to the earthquake shown on the attached image of the
    received alert to the post?; What is the language of received alert?; Does the post include
    the alert contour and user's relative position (YES, NO, NOT_APPLICABLE)?; What is the
    approximate location of the user (shown by blue circle marker on the alert notification)?;
    What type of the alert they receive (BE_AWARE_NOTIFICATION, TAKE_ACTION_ALERT, UNKNOWN)?;
    What is the alert source (AEA, EQN, ETC)?; What is the overall sentiment of the replies to
    the post (CONFIRMATION_OF_POSITIVE_POST, CONFIRMATION_OF_NEGATIVE_POST, OPPOSITION_OF_POSITIVE_POST,
    OPPOSITION_OF_NEGATIVE_POST, NOT_APPLICABLE)?; Did the user feel the earthquake shaking (YES, NO,
    UNKNOWN)?; Did the alert come with a sound notification or was it just a text notification
    (ALERT_WITH_SOUND, SILENT_NOTIFICATION, UNKNOWN)?; What action did the person take after receiving
    the alert (DROP_COVER_HOLD_ON, EVACUATED, MOVED_TO_SAFETY, PROTECTED_OTHERS, PASSIVE_AWARE,
    SOUGHT_INFO, WARNED_CONTACTED_OTHERS, NO_ACTION)?; What is the sentiment of the user (POSITIVE,
    NEGATIVE, NEUTRAL, MIXED)?; Beyond general sentiment, did the user express specific emotions regarding
    the alert or the earthquake (FEAR, ANXIETY, REASSURANCE, GRATITUDE, CONFUSION, SURPRISE,
    ANNOYANCE)?; How helpful or unhelpful was the earthquake alert from the user's point of view
    (NOT_HELPFUL, HELPFUL, VERY_HELPFUL, NEUTRAL)?; Did the user think the system could be improved
    (YES, NO, UNKNOWN)?; When did the alert arrive (BEFORE_SHAKING, DURING_SHAKING, AFTER_SHAKING,
    UNKNOWN)?; What was the level of the shaking that the user felt (STRONG, WEAK, UNKNOWN)?; What
    was the intensity of the ground shaking in MMI scale at the user’s location based on post content
    (1, 2, 3, 4, 5, 6, 7, 8)?; Did someone else near the user receive an earthquake alert as well
    (YES, NO, UNKNOWN)?; Where was the user when received the alert (INDOOR, OUTDOOR, UNKNOWN)?;
    Was the user alone or with others when receiving the alert (YES, NO, UNKNOWN)?; Was it the
    first time the user received an alert from an earthquake alerting system (YES, NO, UNKNOWN)?;
    What was their past experience receiving an earthquake alert (POSITIVE, NEGATIVE, NEUTRAL, UNKNOWN)?;
    Did the user experience earthquake damage in the past (YES, NO, UNKNOWN)?; What was the user’s
    gender (FEMALE, MALE, LIKELY_FEMALE, LIKELY_MALE, UNKNOWN)?; What specific information from the
    alert did the user recall or mention (SAFETY_ADVICE, ESTIMATED_MAGNITUDE, ESTIMATED_DISTANCE,
    ESTIMATED_INTENSITY_AT_THEIR_LOCATION, ALERT_SOURCE like 'Android Earthquake Alerts System)?;
    Did the user comment on the accuracy of the information provided by the AEA (was the magnitude,
    location, or timing perceived as correct or incorrect when compared to their experience or other
    sources, PRECISE, INACCURATE)?; Did the user mention any technical issues with receiving or
    viewing the alert itself (POWER_LOSS, ALERT_SCREEN_FREEZING, ALERT_SOUND_ISSUE, ALERT_NOT_APPEARING_WHEN_EXPECTED,
    UNKNOWN)?; Did the user comment on how clear or easy to understand the alert message and any
    instructions were (CLEAR_TO_UNDERSTAND, ALMOST_CLEAR_TO_UNDERSTAND, UNCLEAR, “UNKNOWN”)?;
    If the user stated they took no specific protective action after receiving the alert did
    they provide a reason why (NO_TIME, CONFUSION, DEEMED_UNNECESSARY, OTHER_REASON_FOR_NO_ACTION,
    REASON_UNSPECIFIED)?; Did the user's post suggest a level of trust (or distrust) in the AEA system
    for future earthquake events based on this particular experience (WILL_TRUST_MORE, WILL_TRUST_LESS,
    NEUTRAL, UNKNOWN)?; If the user explicitly stated the alert was helpful or unhelpful, what specific
    reasons did they give for this assessment (PROVIDED_TIME_TO_PREPARE, CONFIRMED_IT_WAS_AN_EARTHQUAKE,
    IT_ARRIVED_TOO_LATE_TO_BE_USEFUL)?; Did the user compare the Android Earthquake Alert to any other
    earthquake warning systems they might be aware of or other sources of earthquake information
    (AEA_ALERT_ARRIVED_EARLIER, AEA_ALERT_ARRIVED_LATER, AEA_ALERT_WAS_MORE_PRECISE, AEA_ALERT_WAS_LESS_PRECISE,
    UNKNOWN)? If no information is provided for a specific key, use the tag "UNKNOWN" for that key. Make sure to
    limit your response to a JSON format containing only the following keys: “username”, “post_datetime”,
    “post_location”, “warning_time_seconds”, “with_alert_screenshot”, “alert_time”, “magnitude_on_alert_screenshot”,
    “distance_on_alert_screenshot_ml”, “alert_language”, “alert_screenshot_with_contour”, “user_approximate_location_on_alert”,
    “alert_type”, “alert_source”, “reply_sentiment”, “felt_shaking”, “alert_mode”, “post_alert_action”,
    “users_sentiment”, “users_emotion”, “helpfulness”, “system_improvement”, “alert_arrival_wrt_shaking”,
    “shaking_level”, “shaking_intensity_mmi”, “alert_received_by_others”, “indoor_vs_outdoor”, “user's_accompany”,
    “first_earthquake_alert_experience”, “user's_past_earthquake_experience”, “past_earthquake_damage_experience”,
    “user's_gender”, “alert_info_recall”, “aea_info_accuracy”, “technical_issues_with_alert”, “alert_info_clearance”,
    “reason_for_taking_no_action”, “future_trust_level”, “helpfulness_reason”, “aea_vs_others”, “reasoning”.
    Let's walk this through step by step with sample data. Here is the input data: <example_1.png> <example_2.png>.
    Here are the example outputs. <example_1_output.json> <example_2_output.json>
    """


    example_1 = Part.from_uri(
        uri=f"gs://turkey_tweets_0/EXAMPLES/Screenshot 2025-04-24 at 9.58.23 PM.png",
        mime_type="image/png"
    )

    example_1_output = """
    {
    "username": "@cinnamonjemur",
    "post_datetime": "2025-04-23T12:59",
    "post_location": "West of Istanbul",
    "warning_time_seconds": "UNKNOWN",
    "with_alert_screenshot": "YES",
    "alert_time": "2025-04-23T12:49",
    "magnitude_on_alert_screenshot": "5.3",
    "distance_on_alert_screenshot_ml": "88.0",
    "alert_language": "English",
    "alert_screenshot_with_contour": "YES",
    "user_approximate_location_on_alert": "West of Istanbul, near the Marmara Sea coast",
    "alert_type": "BE_AWARE_NOTIFICATION",
    "alert_source": "AEA",
    "reply_sentiment": "CONFIRMATION_OF_POSITIVE_POST",
    "felt_shaking": "YES",
    "alert_mode": "UNKNOWN",
    "post_alert_action": "UNKNOWN",
    "users_sentiment": "POSITIVE",
    "users_emotion": "GRATITUDE",
    "helpfulness": "VERY_HELPFUL",
    "system_improvement": "NO",
    "alert_arrival_wrt_shaking": "UNKNOWN",
    "shaking_level": "UNKNOWN",
    "shaking_intensity_mmi": "UNKNOWN",
    "alert_received_by_others": "UNKNOWN",
    "indoor_vs_outdoor": "UNKNOWN",
    "user's_accompany": "UNKNOWN",
    "first_earthquake_alert_experience": "UNKNOWN",
    "user's_past_earthquake_experience": "UNKNOWN",
    "past_earthquake_damage_experience": "UNKNOWN",
    "user's_gender": "LIKELY_MALE",
    "alert_info_recall": [ "ESTIMATED_MAGNITUDE", "ESTIMATED_DISTANCE", "ALERT_SOURCE" ],
    "aea_info_accuracy": "PRECISE",
    "technical_issues_with_alert": "UNKNOWN",
    "alert_info_clearance": "CLEAR_TO_UNDERSTAND",
    "reason_for_taking_no_action": "UNKNOWN",
    "future_trust_level": "WILL_TRUST_MORE",
    "helpfulness_reason": "CONFIRMED_IT_WAS_AN_EARTHQUAKE",
    "aea_vs_others": "UNKNOWN",
    "reasoning": "The user expresses positive sentiment ('çok iyi' - 'very good') and posts a screenshot of the AEA alert. The date-time of the post was converted from EST to Istanbul time (EST+7). The alert screenshot shows key details like magnitude (5.3) and distance (88.0 miles). The replies confirm the user's positive experience, with another user asking how to enable it and the original poster replying that it works automatically. The user's positive feedback and the nature of the information shared suggest they found the alert helpful and accurate, thereby increasing their trust in the system. Many fields are marked 'UNKNOWN' as the user's short tweet does not provide details on their actions, emotions, or specific experience of the shaking."
    }
    """

    example_2 = Part.from_uri(
        uri=f"gs://turkey_tweets_0/EXAMPLES/Screenshot 2025-05-01 at 1.29.14 PM.png",
        mime_type="image/png"
    )
    example_2_output = """
    {
    "username": "@yigitech",
    "post_datetime": "2025-04-24T07:30",
    "post_location": "Marmara Region",
    "warning_time_seconds": "21",
    "with_alert_screenshot": "YES",
    "alert_time": "UNKNOWN",
    "magnitude_on_alert_screenshot": "4.6",
    "distance_on_alert_screenshot_ml": "41.6",
    "alert_language": "Turkish",
    "alert_screenshot_with_contour": "NO",
    "user_approximate_location_on_alert": "NOT_APPLICABLE",
    "alert_type": "BE_AWARE_NOTIFICATION",
    "alert_source": "AEA",
    "reply_sentiment": "NOT_APPLICABLE",
    "felt_shaking": "YES",
    "alert_mode": "UNKNOWN",
    "post_alert_action": "UNKNOWN",
    "users_sentiment": "POSITIVE",
    "users_emotion": "UNKNOWN",
    "helpfulness": "VERY_HELPFUL",
    "system_improvement": "NO",
    "alert_arrival_wrt_shaking": "BEFORE_SHAKING",
    "shaking_level": "UNKNOWN",
    "shaking_intensity_mmi": "UNKNOWN",
    "alert_received_by_others": "UNKNOWN",
    "indoor_vs_outdoor": "UNKNOWN",
    "user's_accompany": "UNKNOWN",
    "first_earthquake_alert_experience": "UNKNOWN",
    "user's_past_earthquake_experience": "UNKNOWN",
    "past_earthquake_damage_experience": "UNKNOWN",
    "user's_gender": "MALE",
    "alert_info_recall": [ "ESTIMATED_MAGNITUDE", "ALERT_SOURCE" ],
    "aea_info_accuracy": "PRECISE",
    "technical_issues_with_alert": "UNKNOWN",
    "alert_info_clearance": "UNKNOWN",
    "reason_for_taking_no_action": "UNKNOWN",
    "future_trust_level": "WILL_TRUST_MORE",
    "helpfulness_reason": "PROVIDED_TIME_TO_PREPARE",
    "aea_vs_others": "AEA_ALERT_ARRIVED_EARLIER",
    "reasoning": "The user directly compares the performance of Google's Android alert system (AEA) with another application, 'Deprem Ağı' (Earthquake Network). The tweet explicitly states that the Android alert arrived 21 seconds before the earthquake, while the other app's alert arrived 15 seconds before, making AEA faster by 6 seconds. This constitutes a positive sentiment and a direct reason for the alert's helpfulness ('PROVIDED_TIME_TO_PREPARE'). The user provides a screenshot from the 'Deprem Ağı' app, not the AEA alert itself, which is why fields like 'alert_time' are unknown. The post time is estimated based on the earthquake time mentioned in the screenshot plus the 11 minutes mentioned in the tweet text. The user's name 'Yiğit' is male. The distance was converted from km to miles (67km ≈ 41.6 miles)."
    }
    """

    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 0.2,
        "top_p": 0.95,
    }


    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }


    # Automated Process

    blobs1 = storage_client.list_blobs(config.BUCKET, prefix=config.INPUT_PATH_1)
    blobs2 = storage_client.list_blobs(config.BUCKET, prefix=config.INPUT_PATH_2)

    output_dict = {}
    output_filename = "test_results.json"
    output_blob_path = os.path.join(config.OUTPUT_PATH, output_filename)
    blob_to_check = bucket.blob(output_blob_path)

    if blob_to_check.exists():
        print(f"Found existing results file at gs://{config.BUCKET}/{output_blob_path}. Loading...")
        try:
            existing_data = blob_to_check.download_as_string()
            output_dict = json.loads(existing_data)
        except Exception as e:
            print(f"Could not load or parse existing results file. Starting fresh. Error: {e}")
            output_dict = {}

    all_blobs = itertools.chain(blobs1, blobs2)

    for blob in all_blobs:
        if not blob.name.lower().endswith(".png"):
            continue

        file_name = os.path.basename(blob.name)

        if file_name in output_dict:
            print(f"--- Skipping file (already processed): {file_name} ---")
            continue

        print(f"\n--- Processing file: {file_name} ---")
        try:
            image = Part.from_uri(
                uri=f"gs://{config.BUCKET}/{blob.name}",
                mime_type="image/png"
            )

            contents = [
                prompt_instruction,
                example_1,
                example_1_output,
                example_2,
                example_2_output,
                image
            ]

            response = model.generate_content(
                contents,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            # Clean the text, since gemini sometimes uses markdown notation
            cleaned_text = response.text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]

            print("--- Gemini Output Received ---")
            print(cleaned_text)
            print("----------------------------")

            result_data = json.loads(cleaned_text)
            output_dict[file_name] = result_data

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

        time.sleep(1)

    if output_dict:
        try:
            output_blob = bucket.blob(output_blob_path)
            json_string = json.dumps(output_dict, indent=2)
            output_blob.upload_from_string(json_string, content_type="application/json")

            print(f"All results saved to: gs://{config.BUCKET}/{output_blob_path}")
        except Exception as e:
            print(f"Error saving results file: {e}")
    else:
        print("\n---No files were processed.---")


if __name__ == "__main__":
    process_all_tweets()


