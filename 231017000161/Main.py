import calc_adjunct
import calc_adjunct_beta
import calc_adjunct_reduce
import calc_docsim
import calc_radar
import sql_docsim
import sql_utils
from utils_print import print_cross_platform

if __name__ == '__main__':
    rerun = [x for x in range(2, 13)]
    # rerun = []
    taskid = -1
    total = 1343
    print(rerun)
    # record_time = {}
    # record_time[-1]= time.time()

    # step0.
    if 0 in rerun:
        # import sql_labelStore
        # sql_labelStore.insert_labelStore()
        # print_cross_platform('labelStore:insert end')
        # record_time[0] = time.time()
        print_cross_platform('----------------------------------')

    # step1.
    if 1 in rerun:
        # sql_source.push_data('./source/')
        # print_cross_platform('source:insert end')
        # record_time[1] = time.time()
        print_cross_platform('----------------------------------')

    # step2.
    if 2 in rerun:
        calc_docsim.calc_docsim(taskid, total)
        print_cross_platform('docsim:insert + source:update(similarity) end')
        # record_time[2] = time.time()
        print_cross_platform('----------------------------------\t43%\t43%')

    # step3.
    if 3 in rerun:
        sql_docsim.integration(taskid)
        print_cross_platform('docsim:update(similarity mix) end')
        # record_time[3] = time.time()
        print_cross_platform('----------------------------------\t+4%\t47%')

    # step4.
    if 4 in rerun:
        sql_docsim.update_rank(taskid)
        print_cross_platform('docsim:update(rank) end')
        # record_time[4] = time.time()
        print_cross_platform('----------------------------------\t+3%\t50%')

    # step5.
    if 5 in rerun:
        calc_adjunct.calc_adjunct(taskid, total)
        print_cross_platform('adjunct:insert end')
        # record_time[5] = time.time()
        print_cross_platform('----------------------------------\t+33%\t83%')

    # step6.
    if 6 in rerun:
        calc_adjunct_beta.calc_adjunct_beta(taskid)
        print_cross_platform('adjunct_beta:insert end')
        # record_time[6] = time.time()
        print_cross_platform('----------------------------------\t+4%\t87%')

    # step7.
    if 7 in rerun:
        calc_adjunct_reduce.calc_adjunct_reduce(taskid)
        print_cross_platform('adjunct_reduce:insert end')
        # record_time[7] = time.time()
        print_cross_platform('----------------------------------\t+3%\t90%')

    # step8.
    if 8 in rerun:
        calc_adjunct_reduce.calc_adjunct_reduce_score(taskid)
        print_cross_platform('adjunct_reduce:update(scores) end')
        # record_time[8] = time.time()
        print_cross_platform('----------------------------------\t+4%\t94%')

    # step9.
    if 9 in rerun:
        sql_docsim.update_rank(taskid)
        print_cross_platform('docsim:update(rank) end')
        # record_time[9] = time.time()
        print_cross_platform('----------------------------------\t+3%\t97%')

    # step10.
    if 10 in rerun:
        sql_docsim.update_score(taskid)
        print_cross_platform('docsim:update(score) end')
        # record_time[10] = time.time()
        print_cross_platform('----------------------------------\t+1%\t98%')

    # step11.
    if 11 in rerun:
        sql_docsim.update_score_rank(taskid)
        print_cross_platform('docsim:update(scorerank) end')
        # record_time[11] = time.time()
        print_cross_platform('----------------------------------\t+1%\t99%')

    # step12.
    if 12 in rerun:
        calc_radar.calc_radar(taskid)
        print_cross_platform('radar:insert end')
        # record_time[12] = time.time()
        print_cross_platform('----------------------------------\t+1%\t100%')

    sql_utils.close()

    # record_time = eval("{-1: 1550734281.358847, 2: 1550734346.2313468, 3: 1550734353.071347, 4: 1550734357.8563468, 5: 1550734406.1563468, 6: 1550734413.2513468, 7: 1550734419.113847, 8: 1550734426.191347, 9: 1550734431.933847, 10: 1550734432.038847, 11: 1550734434.256347, 12: 1550734436.726347}")
    # sum_time = 0
    # record_time[0] = record_time[-1]
    # record_time[1] = record_time[-1]
    # for i in range(1,len(record_time)):
    #     record_time[len(record_time)-1-i]-=record_time[len(record_time)-1-i-1]
    #     sum_time+=record_time[len(record_time)-1-i]
    # print(record_time)
    # ix=0
    # for i in range(len(record_time)-1):
    #     record_time[i]/=sum_time
    #     record_time[i]=int(record_time[i]*100)
    #     if(record_time[i]==0):
    #         record_time[i]=1
    #     ix+=record_time[i]
    # print(record_time)
    # print(ix)
