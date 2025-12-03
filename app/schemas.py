from pydantic import BaseModel
from typing import Optional

class SentimentRequest(BaseModel):
    text: str

class MusicResponse(BaseModel):
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    similarity_score: float

    class Config:
        from_attributes = True