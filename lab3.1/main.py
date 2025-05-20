import nltk
import numpy
from numpy import dot
from numpy.linalg import norm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def remove_stop_words(tokens):
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend([str(i) for i in range(10)])  # цифры
    stop_words.extend(['.', ',', ':', ';', '!', '?', '–', ')', '(', '\'\'', '\"\"', '``'])
    return [token for token in tokens if token not in stop_words]

def stem_tokens(tokens):
    stemmer = nltk.stem.SnowballStemmer(language='english')
    return [stemmer.stem(token) for token in tokens]

def lem_tokens_eng(tokens):
    lemmer = nltk.stem.WordNetLemmatizer()
    return [lemmer.lemmatize(token) for token in tokens]

def filtered_text(corpus):
    sent_tokens = nltk.sent_tokenize(corpus, language='english')
    new_corpus = []
    for sent in sent_tokens:
        word_tokens = nltk.word_tokenize(sent.lower(), language='english')
        filtered_tokens = remove_stop_words(word_tokens)
        normalized_tokens = lem_tokens_eng(filtered_tokens)
        new_corpus.append(' '.join(normalized_tokens))
    return new_corpus

def get_cosine_similarity(vector_corp, vector_req):
    numpy.seterr(divide='ignore', invalid='ignore')
    return dot(vector_corp, vector_req) / (norm(vector_corp, axis=1) * norm(vector_req))

# Загрузка корпуса
with open('corpus.txt', 'r', encoding='utf-8') as fin:
    corpus = fin.read()

corpus_sent_tokens = nltk.sent_tokenize(corpus, language='english')

# Векторизаторы
bag_of_words = CountVectorizer()
tfidf = TfidfVectorizer()

# Выберите векторизатор: мешок слов или TF-IDF
vectorizer = tfidf  # или bag_of_words

# Предобработка текста
filtered_corpus = filtered_text(corpus)
vector_corpus = vectorizer.fit_transform(filtered_corpus).toarray()

# --- Поисковая система ---
while True:
    request = input("")
    if request.lower() == 'выход':
        print("Работа завершена.")
        break

    filtered_req = filtered_text(request)
    vector_request = vectorizer.transform(filtered_req).toarray()
    similarities = get_cosine_similarity(vector_corpus, vector_request[0])
    best_idx = similarities.argmax()
    if similarities[best_idx] > 0:
        print('Результат:', corpus_sent_tokens[best_idx])
    else:
        print('Совпадений не найдено.')
