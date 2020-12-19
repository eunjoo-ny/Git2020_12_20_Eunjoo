import datetime
now=datetime.datetime.now()
if 3<=now.month<=5:
    print("This month is spring of {}month".format(now.month))
if 6<=now.month<=8:
    print("This month is summer of {}month".format(now.month))
if 9<=now.month<=11:
    print("This month is fall of {}month".format(now.month))
if now.month==12 or 1<=now.month<=2:
    print("This month is winter of{}month".format(now.month))