# G2 Review Feature Extraction with BERT.

# Overview
This project aims to extract customer feature preferences from G2 reviews using a Deep Learning approach centered around BERT (Bidirectional Encoder Representations from Transformers). BERT, being a robust language model, is adept at grasping contextual relationships within text data, rendering it well-suited for the task at hand.

# BERT - (Bidirectional Encoder Representations from Transformers)
  This Python script demonstrates a Deep Learning approach utilizing BERT (Bidirectional Encoder Representations from Transformers) for extracting customer feature preferences from G2 reviews. 
  The script retrieves reviews from the G2 API, preprocesses them, and then uses a fine-tuned BERT model for feature extraction. The extracted feature sets are then printed to the console.

1. Install the required dependencies:
    ```bash
    pip install requests transformers torch
    ```

2. Set up your G2 API access:
    - Obtain a secret token from G2.

3. Run the script:
    ```bash
    python G2_submission.py
    ```

## Script Breakdown

### 1. Data Acquisition

- Utilizes the G2 API with a secret token to retrieve reviews in batches.

### 2. Preprocessing

- Cleans and processes the retrieved review data.

### 3. BERT Model

- Loads a pre-trained BERT model and fine-tunes it on the review dataset.
- Uses the fine-tuned BERT model to extract feature sets from the reviews.

### 4. Review Fetch Function

- Fetches reviews from the G2 API and handles pagination.

### 5. Feature Extraction Function

- Extracts feature sets from the preprocessed reviews using the fine-tuned BERT model.

### 6. Main Function

- Orchestrates the execution of review fetching, preprocessing, feature extraction, and printing of extracted feature sets.

## Customization

- Adjust the API endpoint, parameters, and other configurations as per your requirements.
- Fine-tune the BERT model with your dataset for improved performance.

## Contributors
Rithwik B <br> R Navaneeth Krishnan
