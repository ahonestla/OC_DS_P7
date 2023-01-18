FROM python:3.9-slim as build
EXPOSE 8000
WORKDIR /api/app
COPY /api .
COPY /models/randomforest_v1.pckl ./randomforest_v1.pckl
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt
ENTRYPOINT ["python3", "scoring_api.py"]