# app/Dockerfile
FROM python:3.9-slim
EXPOSE 8501
WORKDIR /app
COPY scoring_app.py ./scoring_app.py
COPY requirements.txt ./requirements.txt
COPY /models/randomforest_v1.pckl ./randomforest_v1.pckl
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit", "run", "scoring_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--theme.base 'dark'"]