
import boto3
import json
import time
from datetime import datetime
import base64

class KinesisDataStream:

    def __init__(self, access_key, secret, region):
       
       self.session = boto3.Session(aws_access_key_id = access_key,
       aws_secret_access_key = secret,
       region_name = region)
       self.kinesis_client = boto3.client("kinesis", region_name="ca-central-1")

    def get_list_streams(self):
        response = self.kinesis_client.list_streams()
        return response
    
    def create_stream(self, stream_name, shard_count=1):
        try:
            response = self.kinesis_client.create_stream(
            StreamName=stream_name,
            ShardCount=1)
            print(f"Kinesis stream '{stream_name}' created successfully!")
        except Exception as e:
           print(f"Error creating stream:{e}")


    def send(self, data, stream_name):


        try:
            json_data = json.dumps({"tweet": data}).encode("utf-8")
            response = self.kinesis_client.put_record(
                StreamName=stream_name,
                Data=json_data,
                PartitionKey="partition-1"
            )
            failed_count = response.get("FailedRecordCount", 0)
            if failed_count > 0:
                    print(f"{failed_count} records failed to send!")
                    for i, record in enumerate(response["Records"]):
                             if "ErrorCode" in record:
                                 print(f"Failed record {i}: {record}")
            print(f"Tweet sent to Kinesis: {data}")
        
        except Exception as e:
            print(f"Error sending data to Kinesis: {e}")


    def send_tweets_from_file(self, stream_name, file_path, batch_size=5):
        """Reads tweets from a file and sends them to Kinesis in batches."""
        try:
            with open(file_path, "r") as file:
                batch = []
                for line in file:
                    tweet_data = json.loads(line.strip())  # Read each line
                    batch.append({
                        'Data': json.dumps(tweet_data).encode("utf-8"),
                        'PartitionKey': 'partition-1'
                    })

                    # Send batch when batch size is met
                    if len(batch) >= batch_size:
                        response = self.kinesis_client.put_records(StreamName=stream_name, Records=batch)
                        print(f"Sent {len(batch)} tweets to Kinesis. Response: {response}")
                        failed_count = response.get("FailedRecordCount", 0)
                        if failed_count > 0:
                            print(f"{failed_count} records failed to send!")
                        batch = []
                        time.sleep(2)  # Prevent AWS throttling

                # Send remaining tweets
                if batch:
                    response = self.kinesis_client.put_records(StreamName=stream_name, Records=batch)
                    print(f"Sent {len(batch)} remaining tweets to Kinesis. Response: {response}")
                    failed_count = response.get("FailedRecordCount", 0)
                    if failed_count > 0:
                       print(f"{failed_count} records failed to send!")
                       for i, record in enumerate(response["Records"]):
                            if "ErrorCode" in record:
                                 print(f"Failed record {i}: {record}")

        except Exception as e:
            print(f"Error sending tweets: {e}")




    def read_from_kinesis(self, stream_name):
       
        try:

            # Get shard ID
            response = self.kinesis_client.describe_stream(StreamName=stream_name)
            shards = response["StreamDescription"]["Shards"]
            if not shards:
                print(f"No shards found in stream '{stream_name}'.")
                return

            for shard in shards:
                shard_id = shard["ShardId"]
                print(f"Using Shard ID: {shard_id}")

                # Get shard iterator (TRIM_HORIZON = start from the oldest record)
                iterator_response = self.kinesis_client.get_shard_iterator(
                    StreamName=stream_name,
                    ShardId=shard_id,
                    ShardIteratorType="TRIM_HORIZON"
                )
                shard_iterator = iterator_response.get("ShardIterator")

                if not shard_iterator:
                    print("No shard iterator found.")
                    continue

                # Continuously fetch records
                while shard_iterator:
                    records_response = self.kinesis_client.get_records(ShardIterator=shard_iterator, Limit=10)
                    print(f"Records response: {records_response}")
                    shard_iterator = records_response.get("NextShardIterator")

                    # Process records
                    for record in records_response.get("Records", []):
                        # Decode the bytes data to a string
                        raw_data = record["Data"].decode("utf-8")
                        print(f"Raw data: {raw_data}")

                        # Parse the JSON string
                        try:
                            tweet_data = json.loads(raw_data)
                            print(f"Received tweet: {tweet_data}")
                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON data: {e}")

                    # Prevent excessive API calls (avoid throttling)
                    time.sleep(1)

        except Exception as e:
            print(f"Error reading from Kinesis: {e}")
            import traceback
            traceback.print_exc()
