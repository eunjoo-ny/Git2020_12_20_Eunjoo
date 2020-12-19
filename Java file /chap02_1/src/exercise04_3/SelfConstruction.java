package exercise04_3;

import java.util.Scanner;
public class SelfConstruction {

	public static void main(String[] args) {
		Scanner scanner=new Scanner(System.in);
		String name;
		int securityNumber;
		int phone;
		
		
		System.out.println("Insert your name:");
		name=scanner.nextLine();
		System.out.println("Input your security number:");
		securityNumber=scanner.nextInt();
		System.out.println("Insert your phone-number:");
		phone=scanner.nextInt();
		
		
		
		System.out.println("1.Your name is "+ name);
		System.out.println("2.Your phone number is "+phone);
		System.out.println("3.Your security number is "+securityNumber);
		
		
		
		

	}

}
