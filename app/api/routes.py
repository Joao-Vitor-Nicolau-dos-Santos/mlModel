from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app import models, schemas
from app.database import get_db
from app.ml_service import SentimentMusicClassifier
import logging

router = APIRouter()
classifier = SentimentMusicClassifier()

# Inicializar o classificador com tratamento de erros
try:
    success = classifier.load_data()
    if success:
        logging.info("Modelo de ML carregado com sucesso")
    else:
        logging.error("Falha ao carregar modelo de ML")
except Exception as e:
    logging.error(f"Erro ao inicializar modelo de ML: {e}")

@router.post("/recommend", response_model=schemas.MusicResponse)
def recommend_music(sentiment: schemas.SentimentRequest, db: Session = Depends(get_db)):
    """
    Recebe um texto de sentimento e retorna uma música recomendada
    """
    try:
        logging.info(f"Recebendo sentimento: {sentiment.text}")
        
        # Verificar se o modelo foi carregado
        if classifier.musics_df is None:
            raise HTTPException(status_code=500, detail="Modelo de ML não está carregado corretamente")
        
        # Usar o classificador de ML para encontrar a música mais apropriada
        recommendations = classifier.predict_music(sentiment.text, top_k=1)
        
        if not recommendations:
            raise HTTPException(status_code=404, detail="Nenhuma música encontrada para o sentimento fornecido")
        
        # Pegar a primeira recomendação
        music_data = recommendations[0]
        
        # Criar objeto de resposta
        response = schemas.MusicResponse(
            id=int(music_data.get('id', 0)),
            title=str(music_data['title']),
            artist=str(music_data['artist']),
            genre=str(music_data['genre']),
            mood=str(music_data['mood']),
            similarity_score=float(music_data['similarity_score'])
        )
        
        logging.info(f"Recomendação: {response}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Erro na recomendação: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Erro interno no sistema de recomendação")

@router.get("/health")
def health_check():
    """Endpoint para verificar saúde da API"""
    model_loaded = classifier.musics_df is not None
    return {"status": "healthy", "ml_model_loaded": model_loaded}