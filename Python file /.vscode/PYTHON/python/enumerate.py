example=["spring","summer","fall","winter"]
print("#simple extraction")
print(example)
print()

print("# the application of enumerate-function ")
print(enumerate(example))
print()

print("#forced change")
print(list(enumerate(example)))


print("#combine the repetition")
for i,value in enumerate(example):

  print("{}th is value:{}^^".format(i,value))