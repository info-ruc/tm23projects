import time

import pymysql

database = pymysql.connect(host="localhost", user="fin",
                           password="guorp", db="financial",
                           port=3306, charset='utf8mb4')
cur = database.cursor()
database_connect_time = time.time()
cnt = 0


def cur_execute(sql):
    # print("=======================================================================")
    # from source import utils_print
    # utils_print.print_cross_platform(sql)
    # print("====================================================================================================================================================")
    try:
        cur.execute(sql)
    except:
        reconnect_manual()
        cur.execute(sql)


def auto_commit():
    global cnt
    cnt += 1
    if cnt & 1023 == 0:
        database.commit()
        cnt = 0


def clear_table(table):
    sql = 'truncate table {table}' \
          ''.format(table=table)
    cur_execute(sql)
    database.commit()


def query_table(table, col_names, condition='1=1'):
    sql = 'select {col_names} from {table} where {condition}' \
          ''.format(col_names=col_names, table=table, condition=condition)
    cur_execute(sql)
    return cur.fetchall()


def clear_table_condition(table, condition):
    sql = 'delete from {table} where {cond}' \
          ''.format(table=table, cond=condition)
    cur_execute(sql)
    database.commit()


def run(sql):
    cur_execute(sql)
    database.commit()


def query_by_sql(sql_text):
    cur_execute(sql_text)
    return cur.fetchall()


def update_by_sql(sql_text):
    cur_execute(sql_text)
    num = cur.rowcount
    database.commit()
    return num


def insert_table_multi(table, data, col_names):
    sql_text = 'insert into {table}({col_names}) values({values})'
    col_names_list = col_names.split(',')
    for (index, line) in enumerate(data):
        values = ','.join([
            '\'{}\''.format(pymysql.escape_string(str(line[col_name])))
            for col_name in col_names_list])
        sql = sql_text.format(table=table, col_names=col_names, values=values)
        cur_execute(sql)
        auto_commit()
    database.commit()


def insert_table_multi_bylist(table, data, col_names):
    sql_text = 'insert into {table}({col_names}) values({values})'
    col_names_list = col_names.split(',')
    for (index, line) in enumerate(data):
        values = ','.join([
            '\'{}\''.format(pymysql.escape_string(str(line[idx])))
            for idx in range(len(col_names_list))])
        sql = sql_text.format(table=table, col_names=col_names, values=values)
        cur_execute(sql)
        auto_commit()
    database.commit()


def insert_table(table, data_line, col_names, commit=True, dev=False):
    sql_text = 'insert into {table}({col_names}) values({values})'
    col_names_list = col_names.split(',')
    values = ','.join([
        '\'{}\''.format(pymysql.escape_string(str(data_line[col_name])))
        for col_name in col_names_list])
    sql = sql_text.format(table=table, col_names=col_names, values=values)
    if dev:
        print(sql)
    cur_execute(sql)
    if commit:
        database.commit()
    else:
        auto_commit()


def update_table_multi(table, data, col_names, conditions):
    sql_text = 'update {table} set {setclause} where {condition}'
    col_names_list = col_names.split(',')
    for (index, line) in enumerate(data):
        setclause = ','.join([
            '{}=\'{}\''.format(col_name, pymysql.escape_string(str(line[col_name])))
            for col_name in col_names_list])
        condition = " and ".join([
            '{}=\'{}\''.format(col_name, pymysql.escape_string(str(conditions[index][col_name])))
            for col_name in conditions[index]])
        sql_temp = sql_text.format(table=table, setclause=setclause, condition=condition)
        cur_execute(sql_temp)
        auto_commit()
    database.commit()


def update_table(table, data_line, col_names, condition, commit=True, dev=False):
    sql_text = 'update {table} set {setclause} where {condition}'
    col_names_list = col_names.split(',')
    setclause = ','.join([
        '{}=\'{}\''.format(col_name, pymysql.escape_string(str(data_line[col_name])))
        for col_name in col_names_list])
    sql = sql_text.format(table=table, setclause=setclause, condition=condition)
    if dev:
        print(sql)
    cur_execute(sql)
    if commit:
        database.commit()
    else:
        auto_commit()


def update_table_statement(table, statement, condition, commit=True, dev=False):
    sql_text = 'update {table} set {statement} where {condition}'
    sql = sql_text.format(table=table, statement=statement, condition=condition)
    if dev:
        print(sql)
    cur_execute(sql)
    if commit:
        database.commit()
    else:
        auto_commit()


def force_commit():
    database.commit()


def close():
    database.close()


def reconnect():
    # global database_connect_time
    # # if time.time() - database_connect_time > 10800: # 3h=3*60*60sec
    # if time.time() - database_connect_time <= 2:  # 2sec
    #     time.sleep(2)
    # database.ping(reconnect=True)
    # database_connect_time = time.time()
    pass


def reconnect_manual(sql=None):
    global database_connect_time, cur, database
    try:
        database.close()
    except:
        pass

    database = pymysql.connect(host="localhost", user="fin",
                               password="luoqs", db="financial",
                               port=3306, charset='utf8mb4')
    cur = database.cursor()
    database_connect_time = time.time()

    if sql is None:
        return None
    else:
        try:
            cur_execute(sql)
            return True
        except:
            return False


def format_resp(data, cols_str):
    cols = cols_str.split(',')
    cols_len = len(cols)
    lst = []
    for line in data:
        dit = {}
        for i in range(cols_len):
            dit[cols[i]] = line[i]
        lst.append(dit)
    return lst


if __name__ == '__main__':
    main_data = [
        {'lclass': 'test11', 'label': 'test21', 'rule': 'test3\'1\'', 'important': '300', 'alias_label': 'test51'}]
    update_table('resume_label_store', main_data, 'lclass,label,rule,important', 'id = 1')
