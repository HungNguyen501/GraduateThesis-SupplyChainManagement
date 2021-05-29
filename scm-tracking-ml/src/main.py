# pylint: disable=no-name-in-module
# pylint: disable=no-self-argument
# uvicorn main:app --reload
from logging import error
import numpy as np
from fastapi import FastAPI, Response, Request
from fastapi.openapi.utils import get_openapi
import re
import json
import requests
from pydantic import BaseModel, Field
from constant import *
import uvicorn
from healthcheck import HealthCheck
import time
from load_model import LSTM, BILSTM, BERT, CheckWrongOrder
from google_translation.translation import TRANSLATION
import warnings
warnings.filterwarnings("ignore")


class Request(BaseModel):
    sign: str = Field(default="", title="log", description="log conatins status description")
    event: str = Field(default="TRACKING_UPDATED", title="log", description="log conatins status description")
    data: dict = Field(default=DEFAULT_INPUT, title="log", description="log conatins status description")


class Respone(BaseModel):
    Message: str = Field(default="Successful", title="log", description="log conatins status description")
    Id: str = Field(default="STOAA00272952YQ", title="log", description="log conatins status description")
    TrackingNumber: str = Field(default="STOAA00272952YQ", title="log", description="log conatins status description")
    wrong_order: int = Field(default=0)
    lstm_score: str = Field(default="TRACKING_ONLINE", title="log", description="log conatins status description")
    lstm_detail: dict = Field(default=DEFAULT_DETAIL, title="log", description="log conatins status description")
    bilstm_score: str = Field(default="TRACKING_ONLINE", title="log", description="log conatins status description")
    bilstm_detail: dict = Field(default=DEFAULT_DETAIL, title="log", description="log conatins status description")
    bert_score: str = Field(default="TRACKING_ONLINE", title="log", description="log conatins status description")
    bert_detail: dict = Field(default=DEFAULT_DETAIL, title="log", description="log conatins status description")  
    TrackLogs: list = Field(default=DEFAULT_TRACK_LOGS, title="log", description="log conatins status description")


class ErrorMessage(BaseModel):
    Message: str = Field(default="Error occured", title="log", description="log conatins status description")
    Id: str = Field(default="STOAA00272952YQ", title="log", description="log conatins status description")
    TrackingNumber: str = Field(default="STOAA00272952YQ", title="log", description="log conatins status description")


def handle_log(TrackLogs):
    """Return handled log and exception log (maybe)
    Parsing tracking logs to handle
    Detect language of log to translate by google
    """
    occur_exception = False
    try:
        list_logs = [(log["c"] + " " + log["d"] + " " + log["z"] + " . ") for log in TrackLogs]

        # Detect language of text
        check_english = True
        for i in range(len(list_logs)):
            list_logs[i] = re.sub("\\d+", " ", list_logs[i])

            if translation.detect_language(list_logs[i]) != "en" and check_english:
                check_english = False

        # Transalte text if language is not English
        if not check_english:
            all_log_to_text = " <google_translate_tab> ".join(list_logs)
            all_log_to_text = translation.translate(all_log_to_text)
            list_logs = all_log_to_text.split("<google_translate_tab>")

            for i in range(len(list_logs)):
                TrackLogs[i]["z"] = list_logs[i]
        
        log_text = "".join(list_logs)
        TrackLogs = TrackLogs[::-1]
        return occur_exception, log_text, TrackLogs

    except Exception as ex:
        occur_exception = True
        return occur_exception, f"exception in handle_log: {str(ex)}", None


if __name__ == "main":  
    lstm = LSTM()
    # bilstm = BILSTM()
    bert = BERT()
    cwo = CheckWrongOrder()
    translation = TRANSLATION()
    health = HealthCheck()
    app = FastAPI()

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Return process time of api in header of response
        """
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

    @app.get("/scm/tracking/ml/healthcheck")
    async def healthcheck():
        """Return of health check score of server
        """
        # return {"status": "200 OK"}
        message, status_code, headers = health.run()
        return json.loads(message) 

    @app.post("/scm/tracking/ml/detect", response_model=Respone, response_model_exclude_unset=True)
    async def detect_status(obj: Request):
        """
        <h3>
        API: <a href="url">http://127.0.0.1:8000/scm/tracking/ml/detect</a> </br></br>
        Method: POST    <br/>
        </h3>
        """  
        error_message = ErrorMessage()
        try: 
            data = obj.data["track"]
            _Id = obj.data["number"]
            _TrackingNumber = obj.data["number"]
            
            error_message.Id = _Id
            error_message.TrackingNumber = _TrackingNumber
            
            if not data:
                error_message.Message = "track is empty"
                return error_message
            elif not (data["z1"] or data["z2"]):
                error_message.Message = "log is empty"
                return error_message

            TrackLogs = None
            if data["z2"]:
                TrackLogs = data["z2"][::-1]
            elif data["z1"]:
                TrackLogs = data["z1"][::-1]

            occur_exception, log_text , TrackLogs = handle_log(TrackLogs)

            if occur_exception:
                error_message.Message = log_text
                return error_message

            response = Respone()
            response.Id = _Id
            response.TrackingNumber = _TrackingNumber
            response.TrackLogs = TrackLogs

            # Predict status
            lstm_pred, lstm_score = lstm.predict(log_text)
            # bilstm_pred, bilstm_score = bilstm.predict(log)
            bert_pred, bert_score = bert.predict(log_text)

            # Check whether order of item is wrong
            cwo_detail, wrong_order_score = cwo.predict(log_text)
            response.wrong_order = wrong_order_score

            # LSTM
            response.lstm_score = lstm_score
            response.lstm_detail = {
                "COMPLETED": lstm_pred[0],
                "DELIVERED_GUARANTEE": lstm_pred[1],
                "IN_US": lstm_pred[2],
                "RETURN_TO_SENDER": lstm_pred[3],
                "TRACKING_AVAILABLE": lstm_pred[4],
                "TRACKING_ONLINE": lstm_pred[5]
            }

            # BILSTM
            response.bilstm_score = "" 
            response.bilstm_detail = DEFAULT_DETAIL

            # BERT
            response.bert_score = bert_score
            response.bert_detail = {
                "COMPLETED": bert_pred[0],
                "DELIVERED_GUARANTEE": bert_pred[1],
                "IN_US": bert_pred[2],
                "RETURN_TO_SENDER": bert_pred[3],
                "TRACKING_AVAILABLE": bert_pred[4],
                "TRACKING_ONLINE": bert_pred[5]
            }
            return response

        except Exception as ex:
            error_message.Message = f"exception orrcured: {str(ex)}"
            return error_message
        
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title="scm-tracking-ml-service",
            version="2.5.0",
            description="Detect status description by Machine Learning",
            routes=app.routes,
        )
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi