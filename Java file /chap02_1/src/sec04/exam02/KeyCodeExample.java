package sec04.exam02;

import java.io.IOException;
public class KeyCodeExample {

	public static void main(String[] args) {
		int intValue;
		System.out.println("Insert your favorite number or character:");
		try {
			
			
				intValue=System.in.read();
			
			System.out.print("Entered number is ");
			int result=(int)intValue;
			System.out.println(result);
		}
		catch(IOException e){
			System.out.println("There exists error. I'm so sorry.");
		}
		
		
		
		
		
		/*int keyCode;
		keyCode=System.in.read();
		System.out.println(keyCode);

		keyCode=System.in.read();
		System.out.println(keyCode);

		keyCode=System.in.read();
		System.out.println(keyCode);*/

	}

}
