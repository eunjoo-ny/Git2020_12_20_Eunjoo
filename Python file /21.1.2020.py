limit=1000
i=0
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
         a = i
         b = j
print("최대가 되는 경우 {}*{}={}".format(a,b,max_value))

print(min([273,52,32,57,103]))



