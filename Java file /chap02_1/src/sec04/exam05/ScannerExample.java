package sec04.exam05;
import java.util.Scanner;

public class ScannerExample {

	public static void main(String[] args) {
		
		Scanner scanner=new Scanner(System.in);
		String A;
		while(true) {
			A=scanner.nextLine();
			System.out.println("Inserted statement:"+A);
			if(A.equals("q")) {
				break;
			}
		}
		System.out.println("I will end the program^^");
		

	}

}
