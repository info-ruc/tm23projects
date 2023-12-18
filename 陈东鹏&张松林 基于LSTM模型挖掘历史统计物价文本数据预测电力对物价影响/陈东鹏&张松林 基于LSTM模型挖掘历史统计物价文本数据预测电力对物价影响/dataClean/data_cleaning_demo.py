
# coding: utf-8

# In[337]:


import pymysql.cursors
import datetime
import pandas as pandas
import numpy as np


# In[338]:


# 【函数说明】：把datetime转成字符串
def datetime_toString(dt):
    return dt.strftime("%Y-%m-%d")

# 【函数说明】：输入dataframe日期升序，日期字段名为date，价格字段名为price
# 填补最小日期到最大日期之间的缺失日期
# 返回：完整的dataframe
def fill_date(dataframe):
    df_plus = pandas.DataFrame() #空表，存储填充的缺失日期序列
    row = dataframe.head(1).copy(); #复制一行作为插入模板
    df_dates = dataframe['date'].tolist()  #数据日期转为列表
    date_start = df_dates[0] #初始时间
    for j in range(0, len(df_dates)):
        date_start_s = datetime_toString(date_start)   #日期转换为字符串类型，使日期可进行逻辑比较
        date_pre_s = datetime_toString(df_dates[j]) #j一直在累加
        # 如数据中日期列表与期望日期序列不相等，即存在缺失值执行while程序
        while (date_pre_s != date_start_s and date_pre_s > date_start_s): #忽略重复日期（最好前期有去重！）
            # print(date_pre_s + " != " + date_start_s + ", add " + date_start_s)
            row.loc[0, ('date',)] = date_start #passes a nested tuple of (slice(None),('one','second'))
            row.loc[0, ('price',)] = None
            df_plus = pandas.concat([df_plus, row]) #将缺失日期新的数据列表中
            date_start += datetime.timedelta(days=1) #日期加一
            date_start_s = datetime_toString(date_start)  
        date_start += datetime.timedelta(days=1) #日期加一
        
    # print("\n df_plus's shape: ", df_plus.shape, "\n", df_plus)
    dataframe = pandas.concat([dataframe, df_plus]) #将缺失日期加入数据列表中（尾部）
    dataframe = dataframe.sort_values(by=['date']) #重新排序
    
    return dataframe

# 【函数说明】：缺失值（价格数据）填充，价格字段名为price
# 返回：待插入的缺失序列（表结构）
def fill_empty_values(dataframe):
    # 方法一：获取缺失值所在行号
    empty_rows = np.where(dataframe.isna())[0] #查找缺失值所在位置
    print("\n缺失值行号：", empty_rows)

    # 缺失值填充：http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.interpolate.html#pandas.DataFrame.interpolate
    dataframe['price'] = dataframe['price'].interpolate('linear') #'linear': ignore the index and treat the values as equally spaced
    # print("\n缺失值填充：\n", dataframe)

    df_insert = pandas.DataFrame() #空表，存储填充的缺失日期序列（已填充缺失值）
    for j in range(0, len(empty_rows)): 
        row_insert = df_filled.iloc[[ empty_rows[j] ]]
        df_insert = pandas.concat([df_insert, row_insert]) #将缺失日期加入数据列表中
    # print("\n待插入的缺失序列：\n", df_insert)

    # 方法二：所有新增行索引为0，需去掉首行（更快，但不如方法一通用）
    # empty_rows_p = df_filled.loc[0]
    # empty_rows = empty_rows_p.iloc[1:len(empty_rows_p)] #查找缺失值所在位置
    # print("\n缺失值行\n：", empty_rows)
    
    return df_insert


# In[339]:


# Connect to the database
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='root',
                             db='farmsdata',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
try:
    with connection.cursor() as cursor:
        # Read a single record
        sql = "SELECT * FROM `apple` WHERE `bazaar` LIKE '%广州%' ORDER BY `date` ASC" #按日期升序，字段名要用倒引号
        cursor.execute(sql)
        result = cursor.fetchall()
        df = pandas.DataFrame(result) #转换成dataframe
        # print("df's shape: ", df.shape, "\n", df.head(30)) #打印前30行
        
        df_filled = fill_date(df)
        print("\n 【缺失日期填充（总表部分）】df_filled's shape:", df_filled.shape, "\n", df_filled.head(30))
        
        df_insert = fill_empty_values(df_filled)
        print("\n 【价格缺失值填充（待插入的缺失序列】df_insert's shape:", df_insert.shape, "\n", df_insert)
        
finally:
    connection.close()


# In[340]:


# 参考：https://blog.csdn.net/leo_sheng/article/details/83316285

