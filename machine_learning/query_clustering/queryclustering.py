from abc import ABC, abstractmethod
from nltk.stem import WordNetLemmatizer
wordnet_lemma = WordNetLemmatizer()
from nltk.corpus import stopwords
from sklearn.cluster import DBSCAN

class Config:
    language = "english"
    eps = 0.35
    min_samples = 1


class QueryClustering(ABC):
    def __init__(self, queries):
        self.queries = queries
        self.config = Config()

    def _remove_stopwords(self, query):
        _query = ""
        for q in query.split():
            if q not in stopwords.words(self.config.language):
                _query += " " + q
        return _query

    def preprocess(self):
        _queries = []
        for query in self.queries:
            _query = self._remove_stopwords(query)
            if len(_query) == 0:
                _query = query
            _query_lemma = wordnet_lemma.lemmatize(word=_query)
            _queries.append(_query_lemma)
        self.processed_queries = _queries

    @abstractmethod
    def metric(self, query_a, query_b):
        pass

    def set_config(self,config):
        self.config = config

    def cluster(self, feature_vector):
        return DBSCAN(metric=self.metric, eps=self.config.eps, min_samples=self.config.min_samples).fit(feature_vector)








