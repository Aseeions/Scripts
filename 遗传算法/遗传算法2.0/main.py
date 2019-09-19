# -*- coding: utf-8 -*-
from genetic import Genetic
from individual import Individual
from group import Group
import matplotlib.pyplot as plt
#from math import sin,cos,tan
import time



if __name__=='__main__':
    t1=time.time()
    gene=Genetic()
    indi=Individual(gene)
    grou=Group(indi)
    
    '''
    gene.genCount=5 #特征数量
    gene.genRange=[[0,100],[0,100],[0,100],[0,100],[0,100]] #特征取值范围
    # fitness=sin(ind[0])-cos(ind[1])+tan(ind[2])+sin(ind[3])*cos(ind[4])#适应度函数
    indi.count=1000 #个体数量
    indi.iterNum=100 #迭代次数
    indi.cross_rate=0.8 #交叉率
    indi.mutation_rate=0.08  #突变率
    '''
    
    pop=grou.initialze(indi.count)   #初始化群体,例[[],[],[],[],[]...]
    print('种群已初始化')
    
    allFitness=grou.AllFitness(pop)    #计算适应度
    print('适应度已计算')
    
    best_fitness=grou.Get_best_fitness(allFitness)   #获取最佳适应度
    best_pop=grou.Get_best_pop(best_fitness)    #获取最佳适应度对应的个体基因
    grou.bestFitness.append(best_fitness)   #将最优适应度放入最优适应度列表
    grou.bestPop=best_pop    ##将最优适应度对应的个体放入最优个体列表
    print('最优已获取')
    
    
    n=0
    iterNum=indi.iterNum
    while n<iterNum:
        print('----------------------------')
        
        pop=indi.Selection(allFitness,pop,grou) #对种群轮盘赌选择
        print('选择已完成')
        print(len(grou.pop))
#        print(len(grou.allFitness))
#        print(len(grou.all_pre_fitness_rate))
#        print('===')
        
        pop=indi.Cross(pop,gene,grou)   #交叉
        print('交叉已完成')
#        print(len(grou.pop))
#        print(len(grou.allFitness))
#        print(len(grou.all_pre_fitness_rate))
#        print('===')
        
        pop=indi.Mutation(pop,gene,grou)    #已有个体的突变放入goup.pop
        print('突变已完成')
#        print(len(grou.pop))
#        print(len(grou.allFitness))
#        print(len(grou.all_pre_fitness_rate))
#        print('===')
        
        new_count=indi.count-len(pop)
        pop=grou.initialze(new_count)   #新产生n个突变个体放入goup.pop
        print('新增已完成')
#        print(len(grou.pop))
#        print(len(grou.allFitness))
#        print(len(grou.all_pre_fitness_rate))
#        print('===')
        
        allFitness=grou.AllFitness(pop)    #计算适应度
        print('适应度已完成')
#        print(len(grou.pop))
#        print(len(grou.allFitness))
#        print(len(grou.all_pre_fitness_rate))
#        print('===')
        
        best_fitness=grou.Get_best_fitness(allFitness)   #获取最佳适应度
        best_pop=grou.Get_best_pop(best_fitness)    #获取最佳适应度个体
        grou.UpdateBest(best_fitness,best_pop)  #更新最佳适应度和最佳个体的基因数据
        print('最优已完成')
#        print(len(grou.bestFitness))
#        print(grou.bestPop) 
#        print(len(grou.pop))
#        print(len(grou.allFitness))
#        print(len(grou.all_pre_fitness_rate))
#        print('===')
        
        n+=1
        print('第%s次迭代已完成' %n)
    t2=time.time()
    print('====================================')
#    print(grou.bestFitness)
    print('总耗时：%0.3f秒' %(t2-t1))
    ind=grou.bestPop
    print('最优个体：%s' %ind)
    print('最佳适应度：%s' %(grou.bestFitness[-1]))
    
#    re=sin(ind[0])-cos(ind[1])+tan(ind[2])+sin(ind[3])*cos(ind[4])
#    print('-------------')
#    print(sin(ind[0]))
#    print(cos(ind[1]))
#    print(tan(ind[2]))
#    print(sin(ind[3]))
#    print(cos(ind[4]))
#    print('-------------')
#    print(re)
    
    x=list(range(indi.iterNum+1))
    plt.figure(figsize=(9,5))
    plt.plot(x,grou.bestFitness)
    plt.show()
    

















