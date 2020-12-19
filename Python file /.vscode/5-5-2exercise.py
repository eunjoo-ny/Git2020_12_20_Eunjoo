#n=0
#items=[]

    #while n<101:
        #n+=1
        #items.append(n)
#from itertools import combinations        
items=['A','B','C','D']

for i in range(1,len(items)):
#print(counter(map(".join",combinations(items,i))))
    print(list(map(".join",itertools.combinations(items,i))))
        
