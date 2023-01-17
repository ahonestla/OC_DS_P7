# app/Dockerfile
FROM python:3.9-slim
EXPOSE 8080
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit", "run", "scoring_app.py", "--server.port=8080", "--server.address=0.0.0.0"]