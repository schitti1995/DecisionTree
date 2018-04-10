import java.util.ArrayList;

//An object of this class represents the properties of the "best" splitting attribute on some data.
class BestAttribute {
    String type;    //nominal or numeric
    double split;   //Splitting value
    int m;          //The attribute column number in the given dataset
    ArrayList<ArrayList<String>> data;  //The discretized data
    
    BestAttribute(String t, double d, int mm, ArrayList<ArrayList<String>> dummy) {
        type = t;
        split = d;
        m = mm;
        this.data = dummy;
    }
}