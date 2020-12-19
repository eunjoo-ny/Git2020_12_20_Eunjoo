
#people=[]

#for j in range(1,100+1):
    #people.append(j)



from itertools import combinations

people=[1,2,3,4,5,6,7,8,9,10]
combi_result=[]
for i in range(1,len(people)):
    for c in combinations(people,i):
        combi_result.append(c)
    print(combi_result)
    