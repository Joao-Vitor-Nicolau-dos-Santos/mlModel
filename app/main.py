from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes
from app.database import engine, Base
import uvicorn
import os

# Criar tabelas no banco de dados
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Sentiment Music Recommender API",
    description="API para recomendação de músicas baseada em sentimentos usando Machine Learning",
    version="1.0.0"
)

# Configurar CORS para permitir requisições do front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, substituir por domínios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rotas
app.include_router(routes.router, prefix="/api/v1", tags=["music-recommendation"])

@app.get("/")
def read_root():
    return {"message": "Sentiment Music Recommender API", "version": "1.0.0"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)