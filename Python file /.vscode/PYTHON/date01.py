import datetime
now=datetime.datetime.now()
A="{}year {}month {}day {}hour {}minute {}second".format(now.year,now.month,now.day,now.hour,now.minute,now.second)
print(A)
print("{}year {}month {}day {}hour {}minute {}second".format(now.year,
    now.month,now.day,now.hour,now.minute,now.second))