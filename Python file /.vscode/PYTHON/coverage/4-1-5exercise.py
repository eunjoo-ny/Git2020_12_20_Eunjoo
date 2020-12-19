number=[1,2,3,4,5,6,7,8,9]
output=[[],[],[]]

for number in numbers:
    #I don't know the reason why "numbers" is error.
   output[(number+2)%3].append(number)

print(output)