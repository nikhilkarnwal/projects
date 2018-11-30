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
        return x>= match_len

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
            if q_b in token_a and token_a[q_b] > 0:
                cmmn+=1
                token_a[q_b]-=1
            else:
                for q_a in token_a.keys():
                    if int(min(len(q_a),len(q_b))) > 5 and token_a[q_a] > 0:
                        if self.lcs_similarity(q_a,q_b,0.75):
                            cmmn+=1
                            token_a[q_a]-=1
        return 1 - cmmn / max(max(len(query_a.split()), len(query_b.split())), 1)

    #function to cluster and return labels of queries in the 
    #input order of queries
    def fit(self):
        feature_vector = np.arange(len(self.processed_queries)).reshape(-1, 1)
        return self.cluster(feature_vector).labels_
