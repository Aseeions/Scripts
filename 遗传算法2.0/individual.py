# -*- coding: utf-8 -*-
from math import sin,cos,tan,floor
import random
from pandas import Series
import numpy as np
from numpy import array


class Individual():
    def __init__(self,genetic):
        self.genetic=genetic
        self.count=100  #个体数
        self.iterNum=200    #染色体进化/迭代次数
        self.fitness=0  #个体的适应度
        self.individual=[]  #个体的基因列表
        self.pre_fitness_rate=0  #个体的累积适应度比率
        self.cross_rate=0.8 #交叉概率
        self.mutation_rate=0.1   #突变概率
        
        
    def initialze(self):
        gene=self.genetic
        geneCount=gene.genCount
        self.individual=[]
        for i in range(geneCount):
            self.individual.append(gene.initialze(i))
        return self.individual
    
    
    def cal_fitness(self,ind):
#        self.fitness=sin(ind[0])-cos(ind[1])+tan(ind[2])+sin(ind[3])*cos(ind[4])
        self.fitness=ind[0]**4-ind[1]*(ind[2]**3)+cos(ind[3])+sin(ind[4])
        return self.fitness
    
    
    def Selection(self,allFitness,pop,group):
        
        #淘汰适应度为负数的个体
        series_fitness=Series(allFitness)
        series_pop=Series(pop)
        cond=series_fitness>0
        allFitness=list(series_fitness[cond])
        pop=list(series_pop[cond])
        print('--已淘汰负数个体')
        
        #计算个体的累积适应度比率
#        group.all_pre_fitness_rate=[]   #恢复初始化状态
        sum_fitness=sum(allFitness)
        fitness_rate=list(map(lambda x:x/sum_fitness,allFitness))
        length=len(allFitness)
        for i in range(length):
            self.pre_fitness_rate=sum(fitness_rate[:i])
            group.all_pre_fitness_rate.append(self.pre_fitness_rate)
        group.all_pre_fitness_rate.append(1)
        print('--已计算累积适应度比率')
        
        #轮盘赌选择
        rand_list=list(np.random.rand(length))  #产生[0，1]随机数，与累计适应度比率比较，确定是否能够存活
#        next_rand=0 #下一个随机数，从头比较适应度
#        next_fit=1  #同一个随机数，比较下一个适应度
#        new_pop=[]  #新个体列表
#        while next_rand<length:
#            if rand_list[next_rand]<=group.all_pre_fitness_rate[next_fit]:
#                new_pop.append(pop[next_fit-1])
#                next_rand+=1
#                next_fit=1
#            else:
#                next_fit+=1
        new_pop=[]
        array_fitness_rate=array(group.all_pre_fitness_rate[1:])
        array_pop=Series(pop)
        for rand_num in rand_list:
            cond=rand_num<=array_fitness_rate   #选择大于随机数的累计适应度比率
            res=list(array_pop[cond])
            new_pop.append(res[0])
        group.pop=new_pop.copy()
        group.allFitness=[]     #适应度列表用不到了，初始化处理
        group.all_pre_fitness_rate=[]   #累积适应度率列表用不到了，初始化处理
        print('--已轮盘赌选择')
        return group.pop
    
    
    def Cross(self,pop,gene,group):
        new_pop=[]
        cutNode=floor(gene.genCount/2)  #基因的交叉点
        length=len(pop)/2
        cross_count=floor(length)
        if cross_count!=length:
            new_pop.append(random.sample(pop,1)[0]) #若群体总数为奇数，则取出一个直接放入下一代群体中

        for i in range(cross_count):
            parents=random.sample(pop,2)
            rand_rate=random.random()
            if rand_rate<=self.cross_rate:
                child_01=parents[0][:cutNode]+parents[1][cutNode:]
                child_02=parents[1][:cutNode]+parents[0][cutNode:]
            else:
                child_01=parents[0]
                child_02=parents[1]
            new_pop.append(child_01)
            new_pop.append(child_02)
        group.pop=new_pop.copy()
        return group.pop
        
                
    def Mutation(self,pop,gene,group):
        #已有个体产生突变
        for i in pop:
            rand_rate=random.random()
            if rand_rate<=self.mutation_rate:
                mut_index=random.randint(0,gene.genCount-1)
                mut_value=random.uniform(gene.genRange[mut_index][0],gene.genRange[mut_index][0])
                i[mut_index]=mut_value
        group.pop=pop.copy()
        return group.pop
    
        
    
    
    
    
    
    
    
    
    
        
        
        
        
        
    
    
    
    
    
            