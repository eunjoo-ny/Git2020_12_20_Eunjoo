limit=10000
i=1
sum=0

while sum<10000:
    sum+=i
    i=i+1
print("if {} is added, then the total-value is more than {} and the exact value is {} ".format(i,limit,sum))