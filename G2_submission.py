import requests
from transformers import BertTokenizer, BertForSequenceClassification
import torch


SECRET_TOKEN = '9b3de46eb58c35663001b93f9f79eaa0447d681671f76250ebcecaae5b8de15d'

#API Endpoint
API_ENDPOINT = 'https://data.g2.com/api/v1/survey-responses'

#Including parameters
params = {
    'filters[product_name]': 'G2 Marketing Solutions',
    'page[size]': 100,
    'page[number]': 1
}

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.eval()

# Feature Sets extraction using BERT
def extract_feature_sets(review_texts):
    feature_sets = []
    for text in review_texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits).item()
        if prediction == 1:  # Assuming label 1 indicates feature request
            feature_sets.append(text)
    return feature_sets

# Review fetch function
def fetch_reviews():
    reviews = []
    try:
        while True:
            response = requests.get(API_ENDPOINT, headers={'Authorization': f'Bearer {SECRET_TOKEN}'}, params=params)
            response.raise_for_status()
            data = response.json()
            if not data.get('data'):
                break
            for review in data['data']:
                review_attributes = review.get('attributes', {})
                title = review_attributes.get('title', '')
                # Only include 'love' and 'hate' comments
                comment_answers = review_attributes.get('comment_answers', {})
                love_comment = comment_answers.get('love', {}).get('value', '')
                hate_comment = comment_answers.get('hate', {}).get('value', '')
                review_content = f"Title: {title}\n"
                if love_comment:
                    review_content += f"Love: {love_comment}\n"
                if hate_comment:
                    review_content += f"Hate: {hate_comment}\n"
                reviews.append(review_content)
            params['page[number]'] += 1
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response content: {response.content}")
    except Exception as err:
        print(f"An error occurred: {err}")
    return reviews

# Main function
def main():
    reviews = fetch_reviews()
    feature_sets = extract_feature_sets(reviews)
    print("Feature Sets Extracted:")
    for idx, feature_set in enumerate(feature_sets, start=1):
        print(f"{idx}. {feature_set}")

if __name__ == "__main__":
    main()
