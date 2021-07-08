import random


class Genetic():
    def __init__(self):
        self.genCount=5 #每条染色体/个体的基因个数
        self.genRange=[[-10,10],[-5,5],[-5,5],[1.5,4.5],[-0.1,3]]  #基因的取值范围
        self.gen=0 #基因数值

    def initialze(self,i):
        self.gen=random.uniform(self.genRange[i][0],self.genRange[i][1])
        return self.gen
    
        