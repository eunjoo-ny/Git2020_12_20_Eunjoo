dictionary_example={"sky":"wind","rain":"cloud","summer":"fall"}
counter={}

for key in dictionary_example:
    
    print(key+":",dictionary_example[key])
    
    

print("#the items' function of dictionary")
print("items():",dictionary_example.items())

print()
print("Combine between the repeated sentence and dictionary")
for key,element in dictionary_example.items():
    print("dictionary[{}]={}".format(key,element))