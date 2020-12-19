class 학생:
    def __init__(self,name,korean,math,english,science):
      self.name= name
      self.koren=korean
      self.math=math
      self.english=english
      self.science=science
    def 총점(self):
      return self.koreanmath+self.math+\
          self.english+self.science
    def 평균(self):
      return self.총점()/4
    def 출력(self):
      print(self.name, self.총점,self.평균,sep="\t")