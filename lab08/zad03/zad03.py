import tweepy
import json
from dotenv import load_dotenv
import os
import time
# Load environment variables from .env file
load_dotenv()

# Twitter API credentials from .env
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")

# Initialize Tweepy Client
client = tweepy.Client(
    consumer_key=API_KEY,
    consumer_secret=API_SECRET,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET
)
client = tweepy.Client(bearer_token=os.getenv("BEARER_TOKEN"))
# Define the search query and number of tweets
query = "NVIDIA"
max_tweets = 100
tweets_per_request = 10  # Adjust to avoid hitting rate limits

# Fetch tweets
tweets = []
paginator = tweepy.Paginator(
    client.search_recent_tweets,
    query=query,
    max_results=tweets_per_request,
    tweet_fields=["text"]
)

for page in paginator:
    tweets.extend(page.data)
    if len(tweets) >= max_tweets:
        break
    time.sleep(1)  # Add delay to avoid rate limits

# Save tweets to a file
with open("tweets.json", "w", encoding="utf-8") as file:
    for tweet in tweets[:max_tweets]:
        json.dump(tweet.data, file, ensure_ascii=False)
        file.write("\n")

print(f"Saved {len(tweets[:max_tweets])} tweets to 'tweets.json'")

# 1 requests / 15 mins
# PER USER