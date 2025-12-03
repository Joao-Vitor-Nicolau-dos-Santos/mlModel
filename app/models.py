from sqlalchemy import Column, Integer, String, Text, Float
from app.database import Base

class Music(Base):
    __tablename__ = "musics"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    artist = Column(String, index=True)
    genre = Column(String)
    mood = Column(String, index=True)  # Sentimento associado
    energy = Column(Float)
    danceability = Column(Float)
    valence = Column(Float)  # Positividade
    tempo = Column(Float)
    acousticness = Column(Float)
    instrumentalness = Column(Float)
    liveness = Column(Float)
    speechiness = Column(Float)