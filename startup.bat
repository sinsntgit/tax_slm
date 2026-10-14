python -m venv venv && venv\Scripts\activate
docker compose up -d
python -m framework.webapp.app --domain tax

