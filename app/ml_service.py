import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import joblib
import os
import logging

# Obter o caminho absoluto da pasta atual (onde está o ml_service.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
DATA_PATH = os.path.join(DATA_DIR, 'music_data.csv')

class SentimentMusicClassifier:
    def __init__(self):
        # Configurando o vetorizador para português
        self.vectorizer = TfidfVectorizer(
            #stop_words='portuguese',
            max_features=1000,
            lowercase=True
        )
        self.model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.musics_df = None
        self.mood_vectors = None
        self.mood_descriptions = None
        
    def load_data(self, data_path=DATA_PATH):
        """Carrega dados de músicas e prepara o modelo"""
        try:
            if not os.path.exists(data_path):
                logging.warning(f"Arquivo {data_path} não encontrado, criando dados de exemplo...")
                self.create_sample_data(data_path)
            
            self.musics_df = pd.read_csv(data_path)
            logging.info(f"Dados carregados: {len(self.musics_df)} músicas")
            
            # Criar representações de sentimentos para cada música (agora em português)
            self.mood_descriptions = self.musics_df.apply(
                lambda x: f"{x['mood']} {x['genre']} {x['artist']} {x['title']}", 
                axis=1
            )
            
            logging.info(f"Descrições de mood: {list(self.mood_descriptions)}")
            
            # Vetorizar os sentimentos
            self.mood_vectors = self.vectorizer.fit_transform(self.mood_descriptions)
            
            # Treinar modelo KNN
            self.model.fit(self.mood_vectors)
            
            logging.info("Modelo de ML carregado com sucesso")
            return True
            
        except Exception as e:
            logging.error(f"Erro ao carregar dados: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def create_sample_data(self, data_path):
        """Cria dados de exemplo para testes (em português)"""
        # Garantir que a pasta 'data' existe
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        sample_data = {
            'id': [1, 2, 3, 4, 5, 6, 7, 8],
            'title': [
                'Música Feliz', 'Blues Triste', 'Rock Energético', 'Jazz Calmo', 
                'Hino Motivacional', 'Balada Romântica', 'Metal Raivoso', 'Piano Pacífico'
            ],
            'artist': [
                'Banda Alegre', 'Alma Melancólica', 'Rockers Poderosos', 'Trio Jazz Suave',
                'Crew Inspiração', 'Corações Amorosos', 'Máquina de Raiva', 'Teclas Tranquilas'
            ],
            'genre': ['Pop', 'Blues', 'Rock', 'Jazz', 'Pop', 'Ballad', 'Metal', 'Clássico'],
            'mood': ['feliz', 'triste', 'animado', 'calmo', 'motivado', 'romântico', 'raivoso', 'pacífico'],
            'energy': [0.8, 0.2, 0.9, 0.3, 0.7, 0.4, 0.95, 0.1],
            'danceability': [0.9, 0.3, 0.6, 0.4, 0.7, 0.5, 0.2, 0.2],
            'valence': [0.9, 0.1, 0.7, 0.6, 0.8, 0.8, 0.3, 0.7],
            'tempo': [120, 60, 140, 80, 110, 70, 160, 75],
            'acousticness': [0.1, 0.8, 0.2, 0.9, 0.3, 0.7, 0.1, 0.95],
            'instrumentalness': [0.1, 0.2, 0.3, 0.7, 0.2, 0.1, 0.8, 0.9],
            'liveness': [0.1, 0.2, 0.8, 0.3, 0.4, 0.2, 0.7, 0.1],
            'speechiness': [0.05, 0.04, 0.08, 0.05, 0.06, 0.05, 0.15, 0.04]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(data_path, index=False)
        logging.info(f"Dados de exemplo criados em {data_path}")
        
    def predict_music(self, sentiment_text, top_k=1):
        """Recebe texto de sentimento e retorna música recomendada"""
        try:
            if self.musics_df is None or self.mood_vectors is None:
                logging.error("Modelo não foi carregado corretamente")
                return []
            
            logging.info(f"Texto de entrada: '{sentiment_text}'")
            
            # Vetorizar o texto de sentimento
            try:
                sentiment_vector = self.vectorizer.transform([sentiment_text])
            except ValueError as e:
                logging.error(f"Erro ao vetorizar texto: {e}")
                # Tentar com uma palavra genérica em português
                sentiment_vector = self.vectorizer.transform(["feliz"])
            
            # Encontrar músicas mais semelhantes usando KNN
            distances, indices = self.model.kneighbors(sentiment_vector, n_neighbors=min(top_k, len(self.musics_df)))
            
            results = []
            for i, idx in enumerate(indices[0]):
                music_data = self.musics_df.iloc[idx].to_dict()
                similarity_score = 1 - distances[0][i]  # Converter distância para similaridade
                
                results.append({
                    **music_data,
                    'similarity_score': float(similarity_score)
                })
                
                logging.info(f"Música encontrada: {music_data['title']} (score: {similarity_score})")
            
            return results
        except Exception as e:
            logging.error(f"Erro na predição: {e}")
            import traceback
            traceback.print_exc()
            return []

# Teste para verificar se o classificador está funcionando
if __name__ == "__main__":
    classifier = SentimentMusicClassifier()
    success = classifier.load_data()
    print(f"Carregamento bem-sucedido: {success}")
    if success:
        print("\nTestando com diferentes sentimentos (em português e inglês):")
        test_cases = [
            "feliz",
            "triste", 
            "animado",
            "calmo",
            "motivado",
            "romântico",
            "raivoso",
            "pacífico",
            "Estou me sentindo feliz e animado hoje",
            "Hoje estou triste e nostálgico",
            "Preciso de música calma para relaxar",
            "Estou apaixonado e romântico",
            "happy",
            "sad",
            "energetic",
            "calm",
            "Palavras aleatórias que não fazem sentido"
        ]
        
        for test_text in test_cases:
            recommendations = classifier.predict_music(test_text, top_k=1)
            if recommendations:
                rec = recommendations[0]
                print(f"  '{test_text}' -> {rec['title']} ({rec['mood']}) - score: {rec['similarity_score']:.3f}")
            else:
                print(f"  '{test_text}' -> Nenhuma recomendação encontrada")