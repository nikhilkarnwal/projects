# Autonomous Query Clustering using DBSCAN

In this post, I am going to talk about Un-supervised Clustering of Syntactically similar queries in python. There are multiple algorithm for clustering with k-means being the popular one but the disadvantage with k-means is that you have to provide k, as number of clusters you want to create. Here , I have used DBSCAN clustering algorithm and LCS distance as pairwise distance between queries.

# DBSCAN (Density-based spatial clustering of applications with noise)
It is a density based clustering algorithm, i.e.  given a set of points in n-dimensional space , it will cluster points that are closely packed together and marking low density region points as outliers. Please check out wiki link for detailed information.

# LCS as Distance Metric
As I need to cluster syntactically similar queries together , I used Longest Common Subsequence algorithm to compute distance between two queries. The big advantage of using this metric is that it take cares of misspelled queries.

    #similarity metric
    def lcs_similarity(self,query_a,query_b,match_len):
        dp=np.array([[0 for i in range(len(query_b)+1)] for y in range(len(query_a)+1)],np.int32)
        for i in range(1,len(query_a)+1):
            for j in range(1,len(query_b)+1):
                if query_a[i-1] == query_b[j-1]:
                    dp[i,j]=dp[i-1,j-1]+1
                else:
                    dp[i,j] = max(dp[i,j-1],dp[i-1,j])
        x= (dp[len(query_a),len(query_b)]/float(max(max(len(query_a),len(query_b)),1)))
        return x&gt;= match_len

Above code is computing the length of the longest common subsequence and returning true if it is greater than the threshold match_len.

</br>
Following is the code in python leveraging nltk library and corpora along with sklearn library for clustering.

# Abstract Class - QueryClustering
This is base class for clustering containing abstract methods metric to be implemented by the class which is inheriting this class. For my purpose, I have created another class SyntacticClustering which is extending QueryClustering and implementing the metric function by using LCS similarity. 

# Functions
 - preprocess() - remove stopwords and provide lemma
 - metric(query_a, query_b) - abstract method to compute distance between two queries
 - clsuter(feature_vector) - cluster feature_vector using metric callable method
 
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

# SyntacticClustering extending QueryClustering
This class is implementing necessary abstract method metric to compute syntactic similarity between queries.

    from queryclustering import QueryClustering
    import numpy as np

    class SyntacticClustering(QueryClustering):

        def __init__(self, queries):
            super().__init__(queries)

        #similarity metric
        def lcs_similarity(self,query_a,query_b,match_len):
            dp=np.array([[0 for i in range(len(query_b)+1)] for y in range(len(query_a)+1)],np.int32)
            for i in range(1,len(query_a)+1):
                for j in range(1,len(query_b)+1):
                    if query_a[i-1] == query_b[j-1]:
                        dp[i,j]=dp[i-1,j-1]+1
                    else:
                        dp[i,j] = max(dp[i,j-1],dp[i-1,j])
            x= (dp[len(query_a),len(query_b)]/float(max(max(len(query_a),len(query_b)),1)))
            return x&gt;= match_len

        #metric to be passed to compute 
        #pairwise distance between queries
        def metric(self, feature_a, feature_b):
            i,j=int(feature_a),int(feature_b)
            query_a, query_b = self.processed_queries[i].lower(), self.processed_queries[j].lower()
            token_a=dict()
            for q_a in query_a.split():
                if q_a in token_a:
                    token_a[q_a]+=1
                else:
                    token_a[q_a]=1
            cmmn=0.0
            for q_b in query_b.split():
                if q_b in token_a and token_a[q_b] &gt; 0:
                    cmmn+=1
                    token_a[q_b]-=1
                else:
                    for q_a in token_a.keys():
                        if int(min(len(q_a),len(q_b))) &gt; 5 and token_a[q_a] &gt; 0:
                            if self.lcs_similarity(q_a,q_b,0.75):
                                cmmn+=1
                                token_a[q_a]-=1
            return 1 - cmmn / max(max(len(query_a.split()), len(query_b.split())), 1)

        #function to cluster and return labels of queries in the 
        #input order of queries
        def fit(self):
            feature_vector = np.arange(len(self.processed_queries)).reshape(-1, 1)
            return self.cluster(feature_vector).labels_

Following is the code to test above module which is passing a file consist of queries and printing labels of each query.

    from syntacticclustering import SyntacticClustering
    from queryclustering import Config
    if __name__ == "__main__":
        query_file = open('abc.txt','r')
        queries=query_file.readlines()

        clst = SyntacticClustering(queries)
        clst.preprocess()
        config = Config()
        config.eps = 0.2
        clst.set_config(config)
        x = clst.fit()
        for i in range(len(x)):
            print(x[i])

        print("done")
