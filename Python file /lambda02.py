A=[1,2,3,4,5,6,7,8,9]
output_a=map(lambda x:x*x,A)
print("the result of embodiment in terms of power:",output_a)
print("the result of embodiment in terms of power:",list(output_a))
output_b=filter(lambda x:3<x<=8,A)
print()
print("the result of realization involved in filter:",output_b)
print("the result of realization involved in filter:",list(output_b))

output_c=map(lambda x:3*x+x*x+x**2,A)
print()
print("the complicated computation:",output_c)
print("the complicated computation:",list(output_c))