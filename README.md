# HungNguyen501-supply-chain-management_graduate-thesis
This project is to serve graduate

<b>Short description:</b>

- The project is to simulate a phase that takes place in E-Commerce. Particularyly, when a order has new event in transport, 17Track Provider updates log to customer (ecommerce company) via Webhook. So, this system gets log from Kafka and puts to ML service to detect status of log to support Customer Service Department.
- Models used to detect status are: LSTM and BERT.

<b>Overview:</b>

System includes 6 main components, in particular:
- <b>scm-tracking-worker</b>: syncs and updates data in real time way from 17Track provider.
- <b>scm-tracking-ml</b>: provides API for detecting the status of log.
- <b>Elasticsearch</b>: stores logs and their status.
- <b>Nifi</b>: handles and migrates data from Elastich search (json type) to Postgres DB (relational data type).
- <b>Grafana</b>: visualizes data to charts like bar chart, line graph, table, .... In addition, it help supervisor monitor data effectively.

This image below is to illustrate:
![scm_tracking_ml_architecture](https://user-images.githubusercontent.com/48648494/121985703-7593cc00-cdbf-11eb-9afb-3473c77d68c1.png)

