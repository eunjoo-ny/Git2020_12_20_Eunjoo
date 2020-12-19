a={
  (0,0): 10,
  (0,1): 20,
  (1,0): 30,
  (1,1): 40
}
print(a[(0,0)])
print(a[1,0])


짝수만= lambda number: number%2==0
a = list(range(100))
b = filter(짝수만,a)
print(list(b))


a=list(range(100))
print(list(map(lambda number: number*number,a)))

file=open("test.txt","a")
file.write("안녕하세요.")
file.close()

with open("test.txt","a") as file:
  file.write("안녕하세요.")

