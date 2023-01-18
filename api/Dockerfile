# app/Dockerfile
FROM python:3.9-slim
EXPOSE 8080
WORKDIR /api/app
COPY /api .
COPY /models/randomforest_v1.pckl ./randomforest_v1.pckl
RUN pip install -r requirements.txt
ENTRYPOINT ["python3", "scoring_api.py"]