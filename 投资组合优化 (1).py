#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install baostock -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn


# In[45]:


#referrence：https://blog.csdn.net/stay_foolish12/article/details/97371586?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522161598796116780357245058%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=161598796116780357245058&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v29-9-97371586.pc_search_result_hbase_insert&utm_term=%E6%8A%95%E8%B5%84%E7%BB%84%E5%90%88%E7%AE%A1%E7%90%86


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco
import scipy.interpolate as sci
from scipy import stats
import baostock as bs
import pandas as pd


# In[ ]:


#通过BaoStock获得上证50成分股股票代码并获取历史数据


# In[4]:


lg = bs.login()


# In[5]:


print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)


# In[6]:


#获取sz50股票代码
rs = bs.query_sz50_stocks()
print('query_sz50 error_code:'+rs.error_code)
print('query_sz50  error_msg:'+rs.error_msg)
sz50_stocks = []
while (rs.error_code == '0') & rs.next():
    sz50_stocks.append(rs.get_row_data())
result = pd.DataFrame(sz50_stocks, columns=rs.fields)
result.to_csv("sz50", encoding="gbk", index=False)
print(result)


# In[7]:


code=result["code"]


# In[ ]:


#获取对应股票近三年价格数据


# In[8]:


history_data = pd.DataFrame(columns = ["date", "code", "close"])
for i in range(0,50):
    data = bs.query_history_k_data_plus(code[i],
    "date,code,close",
    start_date='2018-03-01', end_date='2021-3-01',adjustflag="2")#adjustflag="2"该参数为历史数据前复权
    print('query_history_k_data_plus respond error_code:'+data.error_code)
    print('query_history_k_data_plus respond  error_msg:'+data.error_msg)

    data_list = []
    while (data.error_code == '0') & data.next():
        data_list.append(data.get_row_data())
    result = pd.DataFrame(data_list, columns=data.fields)
    x=result.shape
    print(x)
    if (x[0]==729):
        result.to_csv(code[i], index=0,header=1)
    else:
        code=code.drop([i])#删除近三年价格数据有缺失的股票
    i=i+1
    
    


# In[9]:


code=code.reset_index(drop=True)
print(code.shape)


# In[10]:


#将价格数据合并入一张index为交易日期，columns为股票代码的表格。表格每一列分别代表一只股票的近三年历史收盘价
df=pd.DataFrame(columns=['date','code','close'])
price=pd.DataFrame(np.random.randn(729,43),columns=code)
for i in range(0,43):
    df=df.append(pd.read_csv(code[i]))
    i=i+1  
df=df.reset_index(drop=True)
df.to_csv('df', index=0,header=1) 


# In[11]:


print(df)


# In[12]:


for i in range(0,43):
    for j in range(0,729):
        price.iloc[j,i]=df.iloc[i*729+j,2]
        j=j+1
    i=i+1


# In[13]:


x=pd.DataFrame(columns=['date','code','close'])
x=x.append(pd.read_csv(code[0]))
date=x['date']
print(date)
price=price.set_index(x['date'])
print(price)


# In[ ]:


#将价格股票价格标准化，观察股票趋势


# In[14]:


norm_price = (price/price.iloc[0, :])*100
(norm_price).plot(figsize=(12, 8), grid=True) 


# In[ ]:


#计算股票的对数收益率，将股票当天价格与前一日比较


# In[15]:


log_returns = np.log(price / price.shift(1))
log_returns.head()


# In[ ]:


#股票收益率的直方图，由图可见股票收益率分布近似正态分布


# In[16]:


log_returns.hist(bins=50, figsize=(12, 9))


# In[17]:


'''从柱状图中看出，每只股票收益率分布近似正态分布。
Markowitz均值-方差投资组合理论需要假设正态分布收益率。
而投资组合的风险取决于投资各组合中资产收益率的相关性。
这样，年化收益率和协方差矩阵就是我们需要计算的。
'''


# In[18]:


#使用对数收益率为收益率
rets = log_returns
#计算年化收益率
year_ret = rets.mean() * 729
#计算协方差矩阵
year_volatility = rets.cov() * 729
print(year_ret)


# In[33]:


#我们一共有43支股票，随机生成10000种投资组合。分别计算各投资组合对应的收益率和波动率
number_of_assets = 43

portfolio_returns = []
portfolio_volatilities = []
for p in range (10000):
    weights = np.random.random(number_of_assets)
    weights /= np.sum(weights)
    portfolio_returns.append(np.sum(rets.mean() * weights) * 729)
    portfolio_volatilities.append(np.sqrt(np.dot(weights.T, 
                    np.dot(rets.cov() * 729, weights))))


# In[34]:


portfolio_returns = np.array(portfolio_returns)
portfolio_volatilities = np.array(portfolio_volatilities)


# In[ ]:


#对10000组投资组合的收益率和波动率作散点图


# In[35]:


plt.figure(figsize=(9, 5)) #作图大小
plt.scatter(portfolio_volatilities, portfolio_returns, c=portfolio_returns / portfolio_volatilities, marker='o') #画散点图
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')


# In[36]:


'''
每个点对应某个投资组合，该点有其对应的收益率和波动率（标准差），其颜色为对应的夏普率。可见，越往左上方，夏普率越高。
'''


# In[37]:


def statistics(weights):        
    #根据权重，计算资产组合收益率/波动率/夏普率。
    #输入参数
    #==========
    #weights : array-like 权重数组
    #权重为股票组合中不同股票的权重    
    #返回值
    #=======
    #pret : float
    #      投资组合收益率
    #pvol : float
    #      投资组合波动率
    #pret / pvol : float
    #    夏普率，为组合收益率除以波动率，此处不涉及无风险收益率资产
    #

    weights = np.array(weights)
    pret = np.sum(rets.mean() * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
    return np.array([pret, pvol, pret / pvol])


# In[38]:


#求最大夏普率等于求最小负夏普率，利用scipy.optimize函数求解
def min_func_sharpe(weights):
    return -statistics(weights)[2]
bnds = tuple((0, 1) for x in range(number_of_assets))
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
opts = sco.minimize(min_func_sharpe, number_of_assets 
                    * [1. / number_of_assets,], 
                    method='SLSQP',  bounds=bnds, constraints=cons)#'SLSQP'为Sequential Least Squares Programming方法。
print(opts)


# In[ ]:


#最大夏普率投资组合的权重为：


# In[39]:


opts['x'].round(16)


# In[ ]:


#最大夏普率投资组合的收益率，波动率和夏普率分别为：


# In[40]:


statistics(opts['x']).round(16)


# In[41]:


#定义函数求得在weights比重下投资组合收益率的方差
def min_func_variance(weights):
    return statistics(weights)[1] ** 2
#求最小方差投资组合，利用scipy.optimize函数求解
optv = sco.minimize(min_func_variance, number_of_assets * 
                    [1. / number_of_assets,], method='SLSQP', 
                    bounds=bnds, constraints=cons)
print(optv,statistics(optv['x']).round(16))


# In[42]:


"""
有效边界
有效边界指有效边界上的每一个点即为给定收益率情况下拥有最小波动率的投资组合的点。
"""


# In[44]:


#定义函数求投资组合波动率
def min_func_port(weights):
    return statistics(weights)[1] 
#给定收益率为，从-0.2到0.7之间50等份的值。制造一个线性空间给我们的收益率数组，预设好我们的目标收益率:
target_returns = np.linspace(-0.2, 0.7, 50)
target_volatilities = []
for tret in target_returns:
    cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - tret},
            {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
    res = sco.minimize(min_func_port, number_of_assets * [1. / number_of_assets,], method='SLSQP',
                       bounds=bnds, constraints=cons)
    target_volatilities.append(res['fun'])
    
    #画散点图
plt.figure(figsize=(9, 5))
#圆点为随机资产组合
plt.scatter(portfolio_volatilities, portfolio_returns,
            c=portfolio_returns / portfolio_volatilities, marker='o')
#叉叉为有效边界            
plt.scatter(target_volatilities, target_returns,
            c=target_returns / target_volatilities, marker='x')
#红星为夏普率最大值的资产组合            
plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0],
         'r*', markersize=15.0)
#黄星为最小方差的资产组合            
plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0],
         'y*', markersize=15.0)
            # minimum variance portfolio
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')


# In[ ]:




