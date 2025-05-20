import nltk

nltk.download('punkt_tab')
nltk.download('stopwords') # база стоп-слов 
nltk.download('punkt')	# база пунктуации 
nltk.download('wordnet') # лексическая база данных для англ. яз.

with open('corpus.txt', 'r', encoding='utf-8') as fin: 
    corpus = fin.read()

corpus_sent_tokens = nltk.sent_tokenize(corpus, language='english') 
corpus_word_tokens = nltk.word_tokenize(corpus.lower(), language='english')

print(nltk.corpus.stopwords.words('english'))