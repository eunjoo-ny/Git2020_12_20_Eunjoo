package sec04.exam01;

public class PrintExample {

	public static void main(String[] args) {
		int value=123;
		System.out.printf("the cost of item:%d\n",value);
		System.out.printf("the cost of item:%6d\n",value);
		System.out.printf("the cost of item:%-6d\n",value);
		System.out.printf("the cost of item:%06d\n",value);
		
		double area=10*10*3.14;
		System.out.printf("the area of the circle with radius 10:%10.2f\n",area);
		
        String name="james";
        String job="programmer";
        System.out.printf("%6d:%-10s:%10s\n",1,name,job);
		
		

	}

}
