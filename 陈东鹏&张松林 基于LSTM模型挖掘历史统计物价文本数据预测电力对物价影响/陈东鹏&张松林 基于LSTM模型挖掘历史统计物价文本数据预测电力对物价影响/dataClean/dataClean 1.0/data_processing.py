import tools
import pymysql.cursors


def data_clean(cursor, tableNames):
    for tableName in tableNames:
        tableBazaarList = tools.getBazaarList(tableName, cursor)
        tools.clean_data_with_sql(tableName, tableBazaarList, cursor)
    return


# 参照家辉的封装
def data_summarize(cursor, tableNames):
    for tableName in tableNames:
        tableBazaarList = tools.getBazaarList(tableName, cursor)
        tools.summarized_data_with_sql(tableName, tableBazaarList, cursor)
    return


# @author L-JH ，tableName，想要加哪个表就直接加
tableNames = ['potato']
connection = pymysql.connect(host='localhost',
                             port = 3306,
                             user='root',
                             password='root',
                             db='farmsdata',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

try:
    with connection.cursor() as cursor:
        # print('--------填充缺失日期及空值------------')
        # data_clean(cursor, tableNames)
        print('--------按一定周期汇总数据，取平均值------------')
        data_summarize(cursor, tableNames)

finally:
    connection.close()
