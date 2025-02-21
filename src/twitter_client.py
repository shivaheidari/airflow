import tweepy
import json

class TwitterClient:
    def __init__(self, bearer_token):
        self.client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
    
    def get_real_time_tweets(self, tag,limit=10):

        file_name = tag+"_tweets.json"
        query = f"{tag} -is:retweet lang:en"
        tweets = self.client.search_recent_tweets(query=query, max_results=limit)
        
        
        if tweets.data:
            tweet_list = [tweet.text for tweet in tweets.data]
            with open(file_name, "a") as file:
                for tweet in tweet_list:
                    json.dump({"tweet":tweet}, file)
                    file.write("\n")
            print(f"Saved {len(tweet_list)} tweets locally.")
            return tweet_list
                
        return []
