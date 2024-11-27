import tweepy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import matplotlib.pyplot as plt
import config

bearer_token = config.BEARER_TOKEN

if not bearer_token:
    raise ValueError("BEARER_TOKEN не встановлено в config.py.")

client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

search_query = 'Python -is:retweet lang:en'

max_results = 10
tweet_fields = ['lang', 'text']

try:
    response = client.search_recent_tweets(query=search_query,
                                           max_results=max_results,
                                           tweet_fields=tweet_fields)
except tweepy.TweepyException as e:
    print(f"Помилка під час запиту до API: {e}")
    exit()

tweet_texts = [tweet.text for tweet in response.data] if response.data else []

if not tweet_texts:
    print("Не вдалося знайти твіти за заданим запитом.")
    exit()

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = outputs.logits[0]
    predicted_class = torch.argmax(scores).item()
    if predicted_class == 0:
        return 'Негативний'
    elif predicted_class == 1:
        return 'Позитивний'

sentiments = [analyze_sentiment(text) for text in tweet_texts]

df = pd.DataFrame({
    'Твіт': tweet_texts,
    'Тональність': sentiments
})

print(df)

sentiment_counts = df['Тональніость'].value_counts()
sentiment_counts.plot(kind='bar', color=['red', 'green'])
plt.title('Розподіл тональностей твітів')
plt.xlabel('Тональность')
plt.ylabel('Кількість твітів')
plt.show()
