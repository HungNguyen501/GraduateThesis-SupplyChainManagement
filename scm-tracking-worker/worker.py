import json
from kafka import KafkaConsumer
from json import loads
import requests
from elasticsearch import Elasticsearch
from datetime import datetime


es = Elasticsearch(['http://localhost:9200'])
consumer = KafkaConsumer(
    'scm-tracking-ml',
     bootstrap_servers=['localhost:9092'],
     auto_offset_reset='earliest',
     enable_auto_commit=True,
     group_id='my-group',
     value_deserializer=lambda x: loads(x.decode('utf-8')))


def call_ml_service(body):
    url = "http://127.0.0.1:8000/scm/tracking/ml/detect"
    payload=json.dumps(body)
    headers = {
      '17token': 'A36466A02D9B25A8C99C58CD5AC22414',
      'Content-Type': 'application/json'
    }

    # Send request to ML Service
    response = requests.post(url, headers=headers, data=payload)

    # Check status code
    if response.status_code == 200:
        return response.json()

    return None # Return None if call ML Service unsuccessfully


if __name__ == "__main__":
    for message in consumer:
        data = message.value

        # get time now
        time_now = datetime.now()
        print (f"{time_now} New message")

        # Call ML Service
        ml_response = call_ml_service(data)

        # Parsing ml_respone
        try:
            if ml_response: # 200 OK
                ml_response["update_time"] = time_now
                ml_response["last_log_time"] = ml_response["TrackLogs"][0]["a"]

                # Push ml_response to Elasticsearch to store
                _index = "scm-tracking-ml"
                _id = ml_response["TrackingNumber"]
                _body = ml_response
                res = es.index(index=_index, id=_id, body=_body)
        except Exception as ex:
            print(f"\tException: {str(ex)}")