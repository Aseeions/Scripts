# -*- coding: utf-8 -*-


class Group():
    def __init__(self,individual):
        self.individual=individual
        self.pop=[] #种群的个体列表 
        self.allFitness=[]  #种群所有个体适应度列表
        self.bestFitness=[] #各代最佳适应度列表
        self.bestPop=None #各代中具有最佳适应度的个体列表
        self.all_pre_fitness_rate=[]    #整个种群的累积适应度比率
    
    
    def initialze(self,count):
        indi=self.individual
        for i in range(count):
            self.pop.append(indi.initialze())
        return self.pop
    
    
    def AllFitness(self,pop):
        indi=self.individual
#        self.allFitness=[] #恢复初始化
        for i in pop:
            self.allFitness.append(indi.cal_fitness(i))
        return self.allFitness
    
    
    def Get_best_pop(self,best_fitness):
        best_index=self.allFitness.index(best_fitness)
        best_pop=self.pop[best_index]
        return best_pop
    
    
    def Get_best_fitness(self,pop):
        bestFitness=max(self.allFitness)
        return bestFitness
        
    
    def UpdateBest(self,fitness_new,best_pop):
        self.bestFitness.append(fitness_new)
        bestFitness_last=self.bestFitness[-2]
        bestFitness_new=self.bestFitness[-1]
        if bestFitness_new<bestFitness_last:
            self.bestFitness[-1]=bestFitness_last
        else:
            self.bestPop=best_pop
        
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        