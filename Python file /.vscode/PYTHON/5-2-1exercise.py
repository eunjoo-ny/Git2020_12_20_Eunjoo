def flatten(data):
    result=[]
    for subdata in data:
    
        if type(subdata)==list:
            result+=flatten(subdata)
            
            #for item in subdata:
                # result.append(item)
        else:
            result.append(subdata)
            #if len(data)==0:
                 #result.append(data)
            #else:
                #for item in subdata:
                    # result.append(item)
     return result   
    

example=[[1,2,3],[4,[5,6]],7,[8,9]]
print("original=list:",example)
print("changing list:",flatten(example))

