import numpy as np
from pandas.core.frame import DataFrame
import scipy as scipy
import matplotlib.pyplot as plt
from scipy.optimize.optimize import main
from scipy.stats.morestats import Variance
from scipy.stats.stats import _euclidean_dist
from tabulate import tabulate
import pandas as pd
import statsmodels.api as sm
import pylab as py
import random
from scipy import stats
from xlrd.formula import colname
from fitter import Fitter
from distfit import distfit
import seaborn as sns
from geneview.gwas import manhattanplot
from tqdm import tqdm
from sklearn.svm import SVC

def question1():
    df1= pd.read_csv(r'D:\csds313\Source & Data\airport_routes.csv')
    airport = df1['NumberOfRoutes']
    df2 = pd.read_csv(r'D:\csds313\Source & Data\movie_votes.csv')
    movie = df2['AverageVote']
    fig = sm.qqplot(airport, dist = scipy.stats.distributions.powerlaw(a = 1.6))
    plt.show()
    fig = sm.qqplot(movie, dist = scipy.stats.distributions.norm)
    plt.show()

def question2():
    arr=[]
    for i in range(1000):
        distribution =  []
        normal = np.random.normal(0,1,100)
        for j in range(len(normal)):
            distribution.append(normal[j])
        arr.append(distribution)
    final = list(zip(*arr[::-1]))
    ax = sns.heatmap(final)
    plt.show()
    return final

def question22(final):
    labels = []
    for i in range(100):
        check = final[i][33]+final[i][261]+final[i][425]+final[i][768]+final[i][902]
        if(check>0):
            final[i]=final[i]+(1,)
            labels.append(1)
        else:
            final[i]=final[i]+(0,)
            labels.append(0)
    list1 = []
    pvalue = []
    corralation = []
    for i in range(100):
        list1.append(final[i][1000])
    for i in range(1000):
        list2 = []
        for j in range(100):
            list2.append(final[j][i])
        c,p = stats.pointbiserialr(list1, list2)
        pvalue.append(p)
        corralation.append(c)
    data = [[-np.log10(abs(pvalue[i]))]for i in range(1000)]
    df = pd.DataFrame(data,columns = ['log'])
    df['index']=df.index
    #print(df)
    sns.relplot(
        data = df,
        x = 'index',
        y = 'log',
        aspect = 4,
    )
    #plt.show()
    count = 0
    sig = []
    for i in range(len(pvalue)):
        if(pvalue[i]<=0.01):
            count+=1
            sig.append(i)
    print(sig)
    print(count)

    sig2 = []
    count2 = 0
    for i in range(len(pvalue)):
        if(pvalue[i]<=0.01/1000):
            count2+=1
            sig2.append(i)
    print(sig2)
    print(count2)

    temp = []
    fdr = []
    sig3 = []
    count3 = 0
    import statsmodels
    from sklearn.model_selection import permutation_test_score
    temp,fdr = statsmodels.stats.multitest.fdrcorrection(pvalue) 
    for i in range(len(fdr)):
        if(fdr[i]<=0.1):
            count3+=1
            sig3.append(i)
    print(sig3)
    print(count3)

    clf = SVC(kernel="linear", random_state=7)
    score,perm,p_value =  permutation_test_score(clf,final, labels, n_permutations=1000)
    fig, ax = plt.subplots()
    ax.hist(perm, bins=20, density=True)
    ax.axvline(score, ls="--", color="r")
    #ax.text(0.7, 10, fontsize=12)
    ax.set_xlabel("Accuracy score")
    _ = ax.set_ylabel("Probability")
    plt.show()

    pvalue1 = []
    corralation1 = []
    np.random.shuffle(list1)
    
    for i in range(1000):
        list2 = []
        for j in range(100):
            list2.append(final[j][i])
        c,p = stats.pointbiserialr(list1, list2)
        pvalue1.append(p)
        corralation1.append(c)
    count4 = 0
    sig4 = []
    for i in range(len(pvalue1)):
        if(pvalue1[i]<=0.01):
            count4+=1
            sig4.append(i)
    print(sig4)
    print(count4)

    count5 = 0
    sig5 = []
    for i in range(len(pvalue1)):
        if(pvalue1[i]<=0.001):
            count5+=1
            sig5.append(i)
    print(sig5)
    print(count5)
    
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_distances

def question3(final):
    euclid_dist = []
    cosine_dist = []
    for i in range(len(final)):
        final[i]=list(final[i])
    df = pd.DataFrame(final)
    for i in tqdm(range(100)):
        for j in range(i+1,100):
            euclid_dist.append(distance.euclidean(df.iloc[i,0:1000],df.iloc[j,0:1000]))
    for i in tqdm(range(100)):
        for j in range(i+1,100):
            cosine_dist.append(distance.cosine(df.iloc[i,0:1000],df.iloc[j,0:1000]))  
    fig, ax = plt.subplots()
   # ax.violinplot([euclid_dist])
   # plt.show()
    ax.violinplot([cosine_dist])
    plt.show()
    

def question4(final):
    table = []
    means = []
    variance = []
    dist = []
    for x in range(10):
        euclid_dist = []
        k = 2**x  
        for i in tqdm(range(100)):
            for j in range(i+1,100):
                euclid_dist.append(distance.euclidean(final[i][0:k],final[j][0:k]))
        means.append(np.mean(euclid_dist))
        variance.append(np.var(euclid_dist))
        

        fig, ax = plt.subplots()
        ax.hist(euclid_dist, bins=20, density=True)
        plt.show()

    table.append(means)
    table.append(variance)
    print(table)

def question5():
    from scipy.stats import chi2_contingency
    data = np.array([[213,203,182],[138,110,154]])
    print(chi2_contingency(data)[1])
    from scipy.stats import fisher_exact
    oddsr, p = fisher_exact([[43,17],[5,7]])
    print(p)
    print(chi2_contingency([[43/17],[5/7]])[1])

    
    
   
    
if __name__ == "__main__":
   # final = question2()
    #question22(final)
    #question3(final)
   # question4(final)
    question5()
   # question1()