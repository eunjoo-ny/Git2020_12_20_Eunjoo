#1
list_of_list=[[1,2,3],[4,5,6,7],[8,9]]
for a in list_of_list:
  #2 a=[1,2,3]
  #9 a=[4,5,6,7]
  #18 a=[8,9]
  for b in a:
  #3 b=1
  #4 >1
  #5 b=2
  #6 >2
  #7 b=3
  #8 >3
  #10 b=4
  #11 >4
  #12 b=5
  #13 >5
  #14 b=6
  #15 >6
  #16 b=7
  #17 >7
  #19 b=8
  #20 >8
  #21 b=9
  #22 >9
   print(b)
print("프로그램이 종료되었습니다.")


리스트=[]
리스트=[
    1,
    2,
    3
]

딕셔너리={
    "문자열":"값",
    273:[1,2,3,4],
    True:False}
print(딕셔너리["문자열"])
print(딕셔너리[273])
print(딕셔너리[True])

딕셔너리["문자열"]="값값"
print(딕셔너리["문자열"])

딕셔너리={
    "문자열":"값",
273:[1,2,3,4],
True:False
}
for key in 딕셔너리:
  print("{}:{}".format(key,딕셔너리[key]))

딕셔너리["키"]="값"
print()
for key in 딕셔너리:
  print("{}:{}".format(key,딕셔너리[key]))

del 딕셔너리["키"]
print()
for key in 딕셔너리:
  print("{}:{}".format(key,딕셔너리[key]))

dictionary={
    "name":"7D건조망고",
    "type":"당절임",
    "ingredient":["망고","설탕","메타중아황산나트륨","치자황색소"],
    "origin":"필리핀"
}

print("name:",dictionary["name"])
print("type:",dictionary["type"])
print("ingredient:",dictionary["ingredient"])
print("origin:",dictionary["origin"])
print()

dictionary["name"]="8D 건조망고"
print("name:",dictionary["name"])
#딕셔너리 선언
딕셔너리={
    "문자열":"값",
    273:[1,2,3,4],
    True:False
}
#반복문
for key in 딕셔너리:
  print("{}:{}".format(key,딕셔너리[key]))

#요소추가
딕셔너리["키"]="값"
print()
for key in 딕셔너리:
  print("{}:{}".format(key,딕셔너리[key]))

#요소제거
del 딕셔너리["문자열"]
print()
for key in 딕셔너리:
  print("{}:{}".format(key,딕셔너리[key]))

