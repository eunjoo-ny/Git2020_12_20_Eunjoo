package sec03.exam02;

public class OperationExample {

	public static void main(String[] args) {
    int intValue1=10;
    double intValue2=20;
    double result1=intValue1+intValue2;
    System.out.println(result1);
    
	char charValue1='A';
	char charValue2=1;
	int intValue3=charValue1+charValue2;
	System.out.println("unicode:"+intValue3);
	System.out.println("printed character:"+(char)intValue3);
	
	int doubleValue1=intValue1/4;
	double doubleValue2=intValue1/4.0;
	System.out.println(doubleValue1);
	System.out.println(doubleValue2);
	
	int x=1;
	int y=2;
	float floatValue=(float)x/y;
	System.out.println(floatValue);
	
	
	

	}

}
