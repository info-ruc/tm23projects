import pymysql.cursors
import datetime
import pandas
import numpy


def getBazaarList(tableName, cursor):
    """获取某一品种表下的市场列表

    Disc：
        现在是单品种，单市场。
        下一步，单品种，全市场,L-JH写完,2019-1-13 
        再下一步，多品种，全市场；
        预留多品种的list。
    """
    sql_bazaar = "SELECT `bazaar` FROM `{0}` GROUP BY `bazaar`".format(tableName)
    cursor.execute(sql_bazaar)
    bazaars = cursor.fetchall()
    list = []
    for bazaar in bazaars:
        list.append(bazaar['bazaar'])
    return list


def datetime_toString(dt):
    """把datetime转成字符串

    """
    return dt.strftime("%Y-%m-%d")


def fill_miss_dates_by_asfreq(dataframe):
    """填补缺失日期（!通过pandas.DataFrame.asfreq实现）

    填补最小日期到最大日期之间的缺失日期。日期数据要先去重！！！
    去除重复日期参考：df = df.drop_duplicates(subset='date', keep='first') 

    Args:
        dataframe: 输入dataframe为日期升序。要求其日期字段名为'date'，价格字段名为'price'

    Returns:
        dataframe：完整的dataframe

    Raises:
        IOError: An error occurred accessing the fill_miss_dates_by_asfreq() method.
    """

    dataframe['date'] = pandas.to_datetime(dataframe['date']) 
    dataframe = dataframe.drop_duplicates(['date'])  # 指定列去除重复行
    dataframe = dataframe.set_index('date')  # 将date设置为index

    column_price = dataframe['price'].asfreq(freq='1D')
    dataframe = dataframe.asfreq(freq='1D', method='ffill')
    column_price = column_price.reset_index('date')  # 将date设置的index还原
    dataframe = dataframe.reset_index('date')  # 将date设置的index还原
    dataframe['price'] = column_price['price']

    return dataframe


def fill_miss_dates(dataframe):
    """填补缺失日期（!自己实现，建议用fill_miss_dates_by_asfreq）

    填补最小日期到最大日期之间的缺失日期

    Args:
        dataframe: 输入dataframe为日期升序。要求其日期字段名为'date'，价格字段名为'price'。

    Returns:
        dataframe：完整的dataframe

    Raises:
        IOError: An error occurred accessing the fill_miss_dates() method.
    """

    df_plus = pandas.DataFrame()
    row = dataframe.head(1).copy()  # 复制一行作为插入模板
    df_dates = dataframe['date'].tolist() 
    date_start = df_dates[0]  # 初始时间;前提，日期升序排列
    for j in range(0, len(df_dates)):
        date_start_s = datetime_toString(date_start)   
        date_pre_s = datetime_toString(df_dates[j] ) 
        # 如数据中日期列表与期望日期序列不相等，即存在缺失值执行while程序
        while (date_pre_s != date_start_s and date_pre_s > date_start_s):  # 忽略重复日期（最好前期有去重！）
            # print(date_pre_s + " != " + date_start_s + ", add " + date_start_s)
            row.loc[0, ('date',)] = date_start  # passes a nested tuple of (slice(None),('one','second'))
            row.loc[0, ('price',)] = None
            df_plus = pandas.concat([df_plus, row]) 
            date_start += datetime.timedelta(days=1)  
            date_start_s = datetime_toString(date_start)  
        date_start += datetime.timedelta(days=1) 
        
    # print("\n df_plus's shape: ", df_plus.shape, "\n", df_plus)
    dataframe = pandas.concat([dataframe, df_plus]) 
    dataframe = dataframe.sort_values(by=['date']) 
    
    return dataframe


def fill_empty_values(dataframe):
    """填充缺失值

    填充dataframe中值（价格数据）为NaN的记录。dataframe需先填充缺失日期序列。
    因为需要找到填充行的位置，所以不能调用pandas库函数一步到位（即需先补日期再填充缺失价格）

    Args:
        dataframe: 输入dataframe为日期升序。要求其价格字段名为'price'。

    Returns:
        df_insert：待插入的缺失序列（表结构）

    Raises:
        IOError: An error occurred accessing the fill_empty_values() method.
    """

    # 【方法一】获取缺失值所在行号
    empty_rows = numpy.where(dataframe.isna())[0]
    # print("\n缺失值行号：", empty_rows)  # 19-01-16 修复（不输出）

    dataframe['price'] = dataframe['price'].interpolate('linear')
    # print("\n缺失值填充：\n", dataframe)

    df_insert = pandas.DataFrame()
    for j in range(0, len(empty_rows)): 
        row_insert = dataframe.iloc[[empty_rows[j]]]
        df_insert = pandas.concat([df_insert, row_insert])
    # print("\n待插入的缺失序列：\n", df_insert)

    # 【方法二】若填日期用的是fill_miss_dates，则所有新增行索引为0，需去掉首行（更快，但不如方法一通用）
    # empty_rows_p = dataframe.loc[0]
    # empty_rows = empty_rows_p.iloc[1:len(empty_rows_p)] #查找缺失值所在位置
    # print("\n缺失值行\n：", empty_rows)
    
    return df_insert


def summarized_data_to_mean(time_span='M', df_input=None, product_name='potato', bazaar_name=''):
    """汇总数据并取平均值

    将dataframe中的价格数据按一定时间跨度汇总，计算出每个span的平均值。仅处理单一品种，单一市场。

    Args:
        time_span {
            'M': 按月，"MS"是每个月第一天为开始日期, "M"是每个月最后一天
            “w”：按周，week
            'Q'：按季度，"QS"是每个季度第一天为开始日期, "Q"是每个季度最后一天
            'A'：按年， "AS"是每年第一天为开始日期, "A是每年最后一天
        }

        df: 输入的dataframe。要求其日期字段名为'date'，价格字段名为'price'。
        product_name: 农产品名称
        bazaar_name：市场名称

    Returns:
        df_period：数据类型为dataframe，每行记录一个span和该span的价格平均值。

    Raises:
        IOError: An error occurred accessing the summarized_data_to_mean method.
    """

    df_input['date'] = pandas.to_datetime(df_input['date'])  # 将数据类型转换为日期类型
    df = df_input.set_index('date')  # 将date设置为index
    # print(df.shape)
    # print(df['2015'])

    df_span_mean = df.resample(time_span)['price'].mean()  # df的date需先转DatetimeIndex 
    # print('--------按...的平均数据 row------------')
    # print(df_span_mean)

    # 按月显示，但不统计
    time_span = time_span.strip('S')
    df_period = df_span_mean.to_period(time_span)  # df的date需先转DatetimeIndex
    # print('--------按...的平均数据 remove days------------')
    # print(df_period.head(30))

    df_period = df_period.reset_index('date')  # 将date设置的index还原
    df_period['product_name'] = product_name
    df_period['bazaar_name'] = bazaar_name

    return df_period


def clean_data_with_sql(tableName, bazaarLists, cursor):
    # 按照雪清的方法来,那就是,单市场单品种为基准进行数据清洗
    for bazaar in bazaarLists:
        # 按日期升序，字段名要用倒引号
        sql = "SELECT * FROM `{0}` WHERE `bazaar` LIKE '{1}' ORDER BY `date` ASC".format(tableName, bazaar)
        # sql = "SELECT * FROM `apple` GROUP BY bazaar ORDER BY `date` ASC"
        cursor.execute(sql)
        result = cursor.fetchall()
        if len(result) > 0:
            # 转换成dataframe
            df = pandas.DataFrame(result)
            df = df.drop(columns=['id'])  # 如果有，未加非空判断
            df = df.drop_duplicates(subset='date', keep='first')  # 去除重复日期

            # df_fill_dates = fill_miss_dates(df)
            df_fill_dates = fill_miss_dates_by_asfreq(df)
            print("\n 【缺失日期填充（总表部分）】df_fill_dates's shape:", df_fill_dates.shape, "\n", df_fill_dates.head(30))
            df_insert = fill_empty_values(df_fill_dates)
            print("\n 【价格缺失值填充（待插入的缺失序列】df_insert's shape:", df_insert.shape, "\n", df_insert)

            # 准备插入到数据库... update原则

    return


def summarized_data_with_sql(tableName, bazaarLists, cursor):
    # 按照雪清的方法来,那就是,单市场单品种为基准进行数据清洗
    for bazaar in bazaarLists:
        # 按日期升序，字段名要用倒引号
        sql = "SELECT * FROM `{0}` WHERE `bazaar` LIKE '{1}' ORDER BY `date` ASC".format(tableName, bazaar)
        # 更为细粒度的控制，可以仅查询返回当前周、月、季、年的数据，以免大量重复计算。交给家辉了！

        cursor.execute(sql)
        result = cursor.fetchall()
        if len(result) > 0:
            # 转换成dataframe
            df = pandas.DataFrame(result)
            df = df.drop(columns=['id'])  # 如果有，未加非空判断
            df = df.drop_duplicates(subset='date', keep='first')  # 去除重复日期
           
            df_period_month = summarized_data_to_mean(time_span='MS', df_input=df, product_name=tableName, bazaar_name=bazaar)
            print('--------按月度的平均数据 remove days------------')
            print(df_period_month.head())

            df_period_week = summarized_data_to_mean(time_span='w', df_input=df, product_name=tableName, bazaar_name=bazaar)
            print('--------按周的平均数据 remove days------------')
            print(df_period_week.head())

            df_period_quarter = summarized_data_to_mean(time_span='QS', df_input=df, product_name=tableName, bazaar_name=bazaar)
            print('--------按季度的平均数据 remove days------------')
            print(df_period_quarter.head())

            df_period_year = summarized_data_to_mean(time_span='YS', df_input=df, product_name=tableName, bazaar_name=bazaar)
            print('--------按年的平均数据 remove days------------')
            print(df_period_year.head())

            # 准备插入到数据库... update原则

    return


# 博客参考：
# Python时间序列缺失值处理，https://blog.csdn.net/leo_sheng/article/details/83316285
# Pandas日期数据处理，http://www.mamicode.com/info-detail-1822406.html

# 文档API
# 缺失值填充，http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.interpolate.html#pandas.DataFrame.interpolate


