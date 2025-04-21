from collections import defaultdict
import pandas as pd
import math
import random

class NaiveBayes:

    def __init__(self):
        self.class_probs = defaultdict(float)
        self.word_probs = defaultdict(lambda: defaultdict(float))
        self.classes = ["positive", "negative", "neutral"]  # Ajustado a tus etiquetas

    
    def load_data(self, filepath):
        """Carga específicamente para este dataset de tweets"""
        df = pd.read_csv(filepath, encoding='utf-8')
        
        # Limpieza inicial
        df = df.dropna(subset=['text', 'sentiment'])  # Elimina filas con valores nulos
        df['text'] = df['text'].astype(str)
        
        # Normaliza las etiquetas (el dataset usa 'negative' en lugar de 'negativo')
        label_mapping = {
            'neutral': 'neutral',
            'negative': 'negative', 
            'positive': 'positive'
        }
        df['sentiment'] = df['sentiment'].str.lower().map(label_mapping)
        df = df.dropna(subset=['sentiment'])  # Elimina etiquetas no mapeadas
        
        return df['text'].tolist(), df['sentiment'].tolist()

    def train(self, X, y):
          # Contar frecuencias de palabras por clase
        class_counts = defaultdict(int)
        word_counts = defaultdict(lambda: defaultdict(int))
        
        for tweet, cls in zip(X, y):
            class_counts[cls] += 1
            for word in tweet:
                word_counts[cls][word] += 1

        # Calcular probabilidades
        total_tweets = len(y)
        for cls in self.classes:
            self.class_probs[cls] = class_counts.get(cls, 0) / total_tweets
            total_words = sum(word_counts[cls].values())
            for word in word_counts[cls]:
                self.word_probs[cls][word] = (word_counts[cls][word] + 1) / (total_words + len(word_counts[cls]))


    def predict(self, tweet_tokens):
        # Si no está entrenado, devolver clase aleatoria (para pruebas)
        if not self.class_probs:
            return random.choice(self.classes)
            
        scores = {cls: math.log(self.class_probs[cls]) for cls in self.classes}
        
        for cls in self.classes:
            for word in tweet_tokens:
                if word in self.word_probs[cls]:
                    scores[cls] += math.log(self.word_probs[cls][word])
                else:
                    scores[cls] += math.log(1 / (sum(len(words) for words in self.word_probs.values()) + len(self.word_probs[cls])))
        
        return max(scores.items(), key=lambda x: x[1])[0]