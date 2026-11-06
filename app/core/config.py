from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_NAME: str = "ASMI Consultant API"
    VERSION: str = "1.0.0"
    CORS_ORIGINS: list[str] = [
        o.strip()
        for o in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,https://be-asmi-consultant.vercel.app,https://asmi-consultant.vercel.app",
        ).split(",")
    ]
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    BACKUP_GROQ_API_KEY: str | None = os.getenv("BACKUP_GROQ_API_KEY")

settings = Settings()
