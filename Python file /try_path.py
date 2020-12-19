list=["52","273","32","spy","103"]
list_number=[]
for item in list:
    try:
        float(item)
        list_number.append(item)
    except:
        pass
print("the numbers in {} are".format(list))
print("{}".format(list_number))