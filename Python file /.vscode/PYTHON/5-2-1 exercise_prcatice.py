def new_york(data):
    output=[]
    for subdata in data:
        if type(subdata)==int:
            output.append(subdata)
        elif type(subdata)==list:
            for item in subdata:
                if type(item)==int:
                    output.append(item)
                else:
                    for element in item:
                        output.append(element)
    return output


example=[[1,2,3],[4,[5,6]],7,[8,9]]
print("original=list:",example)
print("changing list:",new_york(example))