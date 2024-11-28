import tweepy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import matplotlib.pyplot as plt
import config

# Отримання токену для доступу до API Twitter з конфігураційного файлу
bearer_token = config.BEARER_TOKEN

# Перевірка наявності токену
if not bearer_token:
    raise ValueError("BEARER_TOKEN не встановлено в config.py.")

# Ініціалізація клієнта для роботи з API Twitter, з обробкою ліміту запитів
client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

# Задання параметрів пошукового запиту
search_query = 'Python -is:retweet lang:en'

# Встановлення максимальної кількості твітів, які необхідно отримати
max_results = 10
tweet_fields = ['lang', 'text']  # Параметри, які потрібно отримати для кожного твіту

# Виконання запиту до API Twitter
try:
    response = client.search_recent_tweets(query=search_query,
                                           max_results=max_results,
                                           tweet_fields=tweet_fields)
except tweepy.TweepyException as e:
    # Виведення повідомлення про помилку, якщо запит не вдався
    print(f"Помилка під час запиту до API: {e}")
    exit()

# Отримання текстів твітів з відповіді
tweet_texts = [tweet.text for tweet in response.data] if response.data else []

# Перевірка наявності твітів у відповіді
if not tweet_texts:
    print("Не вдалося знайти твіти за заданим запитом.")
    exit()

# Завантаження моделі для аналізу тональності тексту
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Завантаження токенізатора
model = AutoModelForSequenceClassification.from_pretrained(model_name)  # Завантаження моделі

# Функція для аналізу тональності одного тексту
def analyze_sentiment(text):
    # Токенізація тексту та передача його в модель
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)  # Отримання передбачення
    scores = outputs.logits[0]  # Логіти (оціночні значення для класів)
    predicted_class = torch.argmax(scores).item()  # Вибір класу з найбільшим значенням
    # Повернення відповідної тональності залежно від класу
    if predicted_class == 0:
        return 'Негативний'
    elif predicted_class == 1:
        return 'Позитивний'

# Аналіз тональності для кожного отриманого твіту
sentiments = [analyze_sentiment(text) for text in tweet_texts]

# Створення таблиці з твітами та їх тональністю
df = pd.DataFrame({
    'Твіт': tweet_texts,
    'Тональність': sentiments
})

# Виведення таблиці з результатами
print(df)

# Підрахунок кількості твітів для кожної тональності
sentiment_counts = df['Тональність'].value_counts()

# Виведення графіку розподілу тональностей
sentiment_counts.plot(kind='bar', color=['red', 'green'])
plt.title('Розподіл тональностей твітів')
plt.xlabel('Тональність')
plt.ylabel('Кількість твітів')
plt.show()  # Виведення графіку на екран
