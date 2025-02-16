from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import boto3
import json

# AWS & API Config
KINESIS_STREAM_NAME = "your-kinesis-stream"
AWS_REGION = "your-region"
ML_API_URL = "http://your_ml_api:5000/predict"

def fetch_kinesis_data():
    """Fetch data from Kinesis Data Stream."""
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

def preprocess_data(**context):
    """Preprocess fetched data before inference."""
    raw_data = context['ti'].xcom_pull(task_ids='fetch_data')
    processed_data = [{"text": d["message"].lower()} for d in raw_data]
    return processed_data

def call_ml_api(**context):
    """Send preprocessed data to ML model API and get predictions."""
    data = context['ti'].xcom_pull(task_ids='preprocess_data')
    response = requests.post(ML_API_URL, json={"data": data})
    predictions = response.json()
    return predictions

def store_results(**context):
    """Store predictions in a database or S3."""
    predictions = context['ti'].xcom_pull(task_ids='call_ml_api')
    # Example: Store in an S3 bucket (modify for DB)
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.put_object(
        Bucket="your-s3-bucket", 
        Key=f"predictions/{datetime.utcnow().isoformat()}.json", 
        Body=json.dumps(predictions)
    )

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 2, 16),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'kinesis_ml_pipeline',
    default_args=default_args,
    description='Fetch data from Kinesis, process it, call ML API, and store results',
    schedule_interval=timedelta(minutes=10),
    catchup=False
)

fetch_task = PythonOperator(
    task_id='fetch_data',
    python_callable=fetch_kinesis_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    provide_context=True,
    dag=dag,
)

api_task = PythonOperator(
    task_id='call_ml_api',
    python_callable=call_ml_api,
    provide_context=True,
    dag=dag,
)

store_task = PythonOperator(
    task_id='store_results',
    python_callable=store_results,
    provide_context=True,
    dag=dag,
)

fetch_task >> preprocess_task >> api_task >> store_task
