import os
import platform

linux = "Windows" not in platform.platform()
interval_num = 0


def print_cross_platform(*args, sep=' ', end='\n', file_name='./log'):
    if linux:
        with open(file_name, 'w', encoding='utf8') as f:
            for item in args:
                f.write(str(item))
                f.write(sep)
            f.write(end)
        os.system('cat log')
    else:
        print(*args, sep=sep, end=end)


def print_skip(*args, sep=' ', end='\n', file_name='./log', interval=2047, print_item="_num"):
    global interval_num
    interval_num += 1
    if interval_num & interval == 0:
        print_cross_platform(*args, sep=sep, end=end, file_name=file_name)
        if (print_item == 'num'):
            print_cross_platform(interval_num)
        return True
    else:
        return False
