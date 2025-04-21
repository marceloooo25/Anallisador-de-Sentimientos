import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

def clean_tweet(tweet):
    """Preprocesamiento optimizado para tweets"""
    if not isinstance(tweet, str):
        return []
    
    # Elimina URLs, menciones, emoticonos y caracteres especiales
    tweet = re.sub(r"http\S+|@\S+|#[A-Za-z0-9_]+|[^\w\sáéíóúñÁÉÍÓÚÑ]", " ", tweet)
    
    # Tokenización avanzada
    tokens = tweet.lower().split()
    stops = set(stopwords.words('english') + ['rt', '...'])
    
    return [token for token in tokens if token not in stops and len(token) > 2]
