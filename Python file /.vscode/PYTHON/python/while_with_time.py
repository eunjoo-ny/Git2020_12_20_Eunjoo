import time

number=0

target_tick=time.time()+5
while target_tick> time.time():
    number+=1
    
print("{} repeat in 5second".format(number))
