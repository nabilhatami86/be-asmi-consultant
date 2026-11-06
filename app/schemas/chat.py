from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

class Msg(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)

class ChatIn(BaseModel):
    messages: list[Msg] = Field(min_length=1, max_length=30)

class ScoutAiIn(BaseModel):
    focus: str = "all"
    gender: str | None = None
    budget_max: float | None = None
    context: str = Field("", max_length=500, description="Konteks tambahan dari user")
