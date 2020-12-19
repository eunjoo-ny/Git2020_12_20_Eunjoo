

#output={2,5,6,11,13,14,23,27,29,30,47,55,59,61,62,95}
output=[i for i in range(1,100+1) if "{:b}".format(i).count("0")==1]


#output=95
#b=format(output,'b')
#print(b)


for i in output:
#print("{:b}".format(output))
    print("{}:{}".format(i,"{:b}".format(i)))
print("total sum=",sum(output))