import datetime

now=datetime.datetime.now()
print("output the current time")
print(now.year,"year")
print(now.month,"month")
print(now.day,"day")
print(now.hour,"hour")
print(now.minute,"minute")
print(now.second,"second")
print()
        
print("output the current time considering format")
output_a=now.strftime("%Y.%m.%D %H:%M:%S")
output_b="{}year {}month {}day {}hour {}minute {}second".format(now.year,\
    now.month,\
        now.day,\
            now.hour,\
                now.minute,\
                    now.second)
output_c=now.strftime("%Y{} %m{} %D{} %H{} %M{} %S{}").format(*"ymdhms")
print(output_a)
print(output_b)
print(output_c)






