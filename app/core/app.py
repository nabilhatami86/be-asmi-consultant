from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import api_router
from app.routers.api import parfum_asmi

def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME, version=settings.VERSION)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_origin_regex=r"https://.*\.vercel\.app|http://localhost:\d+",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root():
        return {
            "status": "online",
            "message": "Parfum ASMI API is running",
            "version": settings.VERSION,
            "docs": "/docs",
        }

    @app.get("/parfum")
    def parfum():
        return parfum_asmi()

    app.include_router(api_router, prefix="/api")
    app.add_api_route("/parfum/asmi", parfum_asmi, methods=["GET"], include_in_schema=False)

    return app

app = create_app()
