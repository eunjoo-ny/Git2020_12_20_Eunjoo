limit=1000
i=1
sum_value=0
while sum_value<limit:
    sum_value+=i
    i+=1
print("{}가 {}를 넘으면 {}이다.".format(i,limit,sum_value))

max_value=0
a=0
b=0
for i in range(1,99+1):
    j=100-i
    print(i,j)
    if max_value< i*j:
        max_value=i*j
        a=i
        b=j
print("최대가 되는 경우 {}*{}={}".format(a,b,max_value))
print(min([273,52,32,57,103]))
print(max([273,52,32,57,103]))
print(sum([273,52,32,57,103]))

a=[0,1,2,3,4,5]
reversed_a=reversed(a)
print(list(reversed_a))
print(list(reversed_a))


a=[0,1,2,3,4,5]
for i in reversed(a):
    print(i,end=" ")
print()
for i in reversed(a):
    print(i,end=" ")

a=[273,103,52,32,57]
print(list(enumerate(a)))


a=[273,103,52,32,57]
for (i,element) in enumerate(a):
    print("{}번째 숫자는 {} 입니다.".format(i,element))

a=[273,103,52,32,57]
for i,element in enumerate(a): #괄호가 없어도 사용가능하므로, 이형태를 기억
    print("{}번째 숫자는 {} 입니다.".format(i,element))


a={"key_1":"value_1","key_2":"value_2","key3":"value_3"}
for key,value in a.items()