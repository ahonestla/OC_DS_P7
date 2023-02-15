FROM python:3.9-slim as build
EXPOSE 8000
WORKDIR /api/app
COPY /api .
COPY /models/xgboost_classifier.pckl ./models/xgboost_classifier.pckl
COPY /models/xgboost_shap_explainer.pckl ./models/xgboost_shap_explainer.pckl
COPY /docs/data_drift_report.html ./docs/data_drift_report.html
COPY /data/processed/test_feature_engineering.gzip ./data/processed/test_feature_engineering.gzip
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt
ENTRYPOINT ["python3", "scoring_api.py"]