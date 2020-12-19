package sec04.exam04;

public class QStopExample {

	public static void main(String[] args)throws Exception {
		System.out.println("Insert the number:");
		int keyCode;
		while(true) {
			keyCode=System.in.read();
			System.out.println("keyCode:"+keyCode);
			
			if(keyCode==113) {
				break;
			}
		}
		System.out.println("End!");
		

	}

}
