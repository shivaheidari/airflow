from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import json
import boto3
import requests
from src.twitter import Twitter
from src.kinesis_datastream import KinesisDataStream

# Configurations
AWS_REGION = "your-region"
KINESIS_STREAM_NAME = "your-kinesis-stream"
ML_API_URL = "http://your_ml_api:5000/predict"
S3_BUCKET_NAME = "your-s3-bucket"

# Initialize Classes
twitter_client = Twitter(api_key="your-twitter-api-key")
kinesis_client = KinesisDataStream(stream_name=KINESIS_STREAM_NAME)

def fetch_tweets():
    """Fetch tweets using the Twitter class."""
    tweets = twitter_client.get_tweets()  # Returns a list of tweet dicts
    return tweets

def send_to_kinesis(**context):
    """Send tweets to Kinesis."""
    tweets = context['ti'].xcom_pull(task_ids='fetch_tweets')
    for tweet in tweets:
        kinesis_client.send(tweet)  # Sends tweet to Kinesis

def read_from_kinesis():
    """Read tweets from Kinesis Data Stream."""
    return kinesis_client.read()

def call_ml_api(**context):
    """Send tweets to ML API for emotion prediction."""
    tweets = context['ti'].xcom_pull(task_ids='read_from_kinesis')

    predictions = []
    for tweet in tweets:
        response = requests.post(ML_API_URL, json={"tweet": tweet["text"]})
        prediction = response.json()
        predictions.append({"tweet": tweet["text"], "prediction": prediction["pred"]})
    
    return predictions

def store_in_s3(**context):
    """Save predictions in S3, creating bucket if necessary."""
    predictions = context['ti'].xcom_pull(task_ids='call_ml_api')
    s3 = boto3.client("s3", region_name=AWS_REGION)

    # Create bucket if it doesn't exist
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
    'twitter_kinesis_pipeline',
    default_args=default_args,
    description='Fetch tweets, stream to Kinesis, predict emotion, and save to S3',
    schedule_interval=timedelta(minutes=10),
    catchup=False
)

fetch_task = PythonOperator(task_id='fetch_tweets', python_callable=fetch_tweets, dag=dag)
send_task = PythonOperator(task_id='send_to_kinesis', python_callable=send_to_kinesis, provide_context=True, dag=dag)
read_task = PythonOperator(task_id='read_from_kinesis', python_callable=read_from_kinesis, dag=dag)
predict_task = PythonOperator(task_id='call_ml_api', python_callable=call_ml_api, provide_context=True, dag=dag)
store_task = PythonOperator(task_id='store_in_s3', python_callable=store_in_s3, provide_context=True, dag=dag)

fetch_task >> send_task >> read_task >> predict_task >> store_task

