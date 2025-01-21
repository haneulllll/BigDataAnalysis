import java.util.Vector;

public class Main{
    public static void main(String[] args){
        Product[] items = new Product[3];
        items[0] = new Product(10, "!!!");
        items[1] = new TV("TV");
        items[2] = new Product(20, "!@#");
        for(int i = 0; i < items.length; i++){
            System.out.println(items[i].price + items[i].name);
        }
    }
}

class Product{
    int price;
    String name;

    Product(int price, String name){
        this.price = price;
        this.name = name;
    }
}

class TV extends Product{
    TV(String name){
        super(5000, name);
    }
}

class myex extends Exception{
    private final int ERRcode;
    myex(String msg, int errCode){
        super(msg);
        ERRcode = errCode;
    }
    myex(String msg){
        this(msg, 100);
    }

    int getErrcode(){
        return ERRcode;
    }
}