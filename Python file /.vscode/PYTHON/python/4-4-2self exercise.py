source=[i for i in range(1,1000+1) if "{:x}".format(i).count("4")==2]
        
for i in source:
    print("{}:{}".format(i,"{:x}".format(i)))
print("total sum=",sum(source))
                            