from itertools import combinations

relations=[["100","ryan","music","2"],
           ["200","apeach","math","2"],
           ["300","tube","computer","3"],
           ["400","con","computer","4"],
           ["500","muzi","music","3"],
           ["600","apeach","music","2"]]
result=[]
for element in relations:
    for i in range(1,len(element)):
        for c in combinations(element,i):
            result.append(c)
print(result)
print(result.count)
           

