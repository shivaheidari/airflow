from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import tweepy
import boto3
import json
import requests

# Configurations
TWITTER_BEARER_TOKEN = "your-twitter-bearer-token"
KINESIS_STREAM_NAME = "your-kinesis-stream"
AWS_REGION = "your-region"
ML_API_URL = "http://your_ml_api:5000/predict"
S3_BUCKET_NAME = "your-s3-bucket"

def fetch_tweets():
    """Fetch tweets from Twitter API and send to Kinesis."""
    client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
    kinesis = boto3.client("kinesis", region_name=AWS_REGION)

    query = "AI OR machine learning OR data science -is:retweet"
    tweets = client.search_recent_tweets(query=query, max_results=10, tweet_fields=["created_at", "text"])

    for tweet in tweets.data:
        data = json.dumps({"id": tweet.id, "text": tweet.text, "timestamp": tweet.created_at.isoformat()})
        kinesis.put_record(StreamName=KINESIS_STREAM_NAME, Data=data, PartitionKey="partition-1")

def read_kinesis():
    """Read tweets from Kinesis Data Stream."""
    kinesis = boto3.client("kinesis", region_name=AWS_REGION)
    response = kinesis.get_shard_iterator(
        StreamName=KINESIS_STREAM_NAME, 
        ShardId='shardId-000000000000', 
        ShardIteratorType='LATEST'
    )
    iterator = response['ShardIterator']
    records_response = kinesis.get_records(ShardIterator=iterator, Limit=10)

    data = [json.loads(record['Data']) for record in records_response['Records']]
    return data

def call_ml_api(**context):
    """Call ML API to predict emotion of tweets."""
    tweets = context['ti'].xcom_pull(task_ids='read_kinesis')
    
    responses = []
    for tweet in tweets:
        response = requests.post(ML_API_URL, json={"text": tweet["text"]})
        prediction = response.json()
        responses.append({"id": tweet["id"], "text": tweet["text"], "prediction": prediction})
    
    return responses

def save_to_s3(**context):
    """Save predictions to S3, creating the bucket if needed."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    predictions = context['ti'].xcom_pull(task_ids='call_ml_api')

    # Create bucket if it does not exist
    try:
        s3.head_bucket(Bucket=S3_BUCKET_NAME)
    except:
        s3.create_bucket(Bucket=S3_BUCKET_NAME)

    filename = f"predictions/{datetime.utcnow().isoformat()}.json"
    s3.put_object(Bucket=S3_BUCKET_NAME, Key=filename, Body=json.dumps(predictions))

# Airflow DAG configuration
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 2, 16),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'twitter_emotion_pipeline',
    default_args=default_args,
    description='Fetch tweets, predict emotions, and save to S3',
    schedule_interval=timedelta(minutes=10),
    catchup=False
)

fetch_task = PythonOperator(task_id='fetch_tweets', python_callable=fetch_tweets, dag=dag)
read_task = PythonOperator(task_id='read_kinesis', python_callable=read_kinesis, provide_context=True, dag=dag)
predict_task = PythonOperator(task_id='call_ml_api', python_callable=call_ml_api, provide_context=True, dag=dag)
save_task = PythonOperator(task_id='save_to_s3', python_callable=save_to_s3, provide_context=True, dag=dag)

fetch_task >> read_task >> predict_task >> save_task
