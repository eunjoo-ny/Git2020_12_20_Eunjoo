while True:
  string_input=input("input integer")
  if string_input.isdigit():
        number_input_a=int(string_input)
        print("radius of circle:",number_input_a)
        print("circumstance of circle:",2*3.14*number_input_a)
        print("area of circle:",3.14*number_input_a*number_input_a)
        break
  else:
        print("please input integer")