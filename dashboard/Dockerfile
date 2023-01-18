# app/Dockerfile
FROM python:3.9-slim
EXPOSE 8501
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit", "run", "scoring_app.py", "--theme.base", "dark", "--server.port=8501", "--server.address=0.0.0.0"]