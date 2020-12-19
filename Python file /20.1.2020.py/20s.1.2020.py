i=0
while i<10:
    print("{}번째 반복문입니다.".format(i))
    i+=1

 #변수를 선언합니다.
list_test=[1,2,1,2]
value=2
#list_test 내부에 value 가 있다면 반복
while value in list_test:
    list_test.remove(value)
print(list_test)

#시간과 관련된 기능을 가져온다.
import time
#변수를 선언합니다.
number=0
#5초동안반복하겠다. target_tick
target_tick= time.time()+5
while time.time() < target_tick:
    number+=1
print("5초 동안 {}번 반복했습니다.".format(number))

#변수를 선언합니다.
i=0
#무한반복을 실행합니다.
while True:
    print("{}번째 반복문입니다.".format(i))
    #몇번째인지를 실행한다.
    i+=1
    #반복문을 종료하기 위한 break
    input_text=input("' >종료하시겠습니까?(y):")
    if input_text in ["y","Y"]:
        print("반복을 종료합니다.")
        break

numbers=[5,15,6,20,7,25]
for number in numbers:
    if number <10:

        print(number)


numbers=[5,15,6,20,7,25]
for number in numbers:
    if number <10:
        continue
        print(number)

