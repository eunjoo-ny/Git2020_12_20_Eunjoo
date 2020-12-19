p=0
q=0
max=0
for i in range(1,100):
    j=100-i
    
    if max<i*j:
        max=i*j
        p=i
        q=j
print("In the case of maximum:{}*{}={}".format(p,q,max))