import nltk
import pymorphy2
import numpy 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import dot
from numpy.linalg import norm

with open('corpus.txt', 'r', encoding='utf-8') as fin: corpus = fin.read()

def remove_stop_words(tokens):
    stop_words = nltk.corpus.stopwords.words('english') 
    stop_words.extend(range(10)) 
    stop_words.extend(['.', ',', ':', ';', '!', '?', '–', ')', '(',
                       '\'\'', '\"\"', '``'])
    return [token for token in tokens if token not in stop_words]

def stem_tokens(tokens):
    stemmer = nltk.stem.SnowballStemmer(language='english') 
    return [stemmer.stem(token) for token in tokens]

def lem_tokens_eng(tokens):
    lemmer = nltk.stem.WordNetLemmatizer()
    return [lemmer.lemmatize(token) for token in tokens]

def filtered_text(corpus):
    sent_tokens = nltk.sent_tokenize(corpus, language='english') 
    new_corpus = [ ]
    for sent in sent_tokens:
        word_tokens = nltk.word_tokenize(sent.lower(), language='english') 
        filtered_tokens = remove_stop_words(word_tokens) # удаление стоп-слов print(filtered_tokens)
        normalized_tokens = lem_tokens_eng(filtered_tokens)# лемматизация
        new_corpus.append(' '.join(normalized_tokens))
    return new_corpus

def get_cosine_similarity(vector_corp, vector_req): 
    numpy.seterr(divide='ignore', invalid='ignore')
    return dot(vector_corp, vector_req) / (norm(vector_corp, axis=1) *
                                            norm(vector_req))

corpus_sent_tokens = nltk.sent_tokenize(corpus, language='english') 
corpus_word_tokens = nltk.word_tokenize(corpus.lower(), language='english')    

bag_of_words = CountVectorizer() 
tfidf = TfidfVectorizer()    
vectorizer = bag_of_words
vector_corpus = vectorizer.fit_transform(filtered_text(corpus)).toarray()


while True:
    request = input('Введите запрос:... (или выход для завершения)')
    if request.lower() == 'выход':
        print('Поисковик закрыт')
        break

    vector_request = vectorizer.transform(filtered_text(request)).toarray()
    response_id = get_cosine_similarity(vector_corpus, vector_request[0]).argmax()    
     
    if numpy.all(vector_request == 0):
        print('Результат не найден')
    else:
        print('Result: ' + corpus_sent_tokens[response_id])


     
    
        
    
        