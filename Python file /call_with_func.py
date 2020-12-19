def power(item):
    return item*item
def less_3(item):
    return item<=3
def more_2(item):
    return 5>=item>=2
def multi(item):
    return item**2+3*item

A=[1,2,3,4,5,6,7,8,9]
output_a=map(power,A)
print("embodiment of a map-function:")
print("map(power,A)",map(power,A))
print("map(power,A)",list(map(power,A)))
print()
output_b=filter(less_3,A)
print("realization of a filter-function:")
print("filter(less_3,A)",filter(less_3,A))
print("filter(less_3,A)",list(filter(less_3,A)))
print()
output_c=filter(more_2,A)
print("realization of a filter-function:")
print("filter(less_3,A)",output_c)
print("filter(less_3,A)",list(output_c))
print()
output_d=map(multi,A)
print("embodiment of a map-function:")
print("map(multi,A)",output_d)
print("map(multi,A)",list(output_d))
    