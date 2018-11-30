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
