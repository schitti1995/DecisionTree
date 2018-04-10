import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import weka.core.Instance;
import weka.core.Instances;

class AuxMethods {
    /**
     * This method prints the pre-order traversal of the given decision tree.
     * @param root
     */
    public static void printTree(TreeNode root) {
        printTree(root, 0);
    }
    
    
    /**
     * Auxiliary method used for printing the tree.
     * @param root
     * @param level
     */
    private static void printTree(TreeNode root, int level) {
        for(int i = 0; i <= level; i++)
            System.out.print("| ");
        
        if(root.isLeaf) {
            System.out.println("Label = " + root.label);
            return;
        }
        
        System.out.println(decisionTree.attributes.get(root.attribute.m) + " ---> split value at parent node: " + root.prev_SplitVal);
        for(TreeNode child : root.adj)
            printTree(child, level+1);
        return;
    }
    
    
    /**
     * This method reads the data from the WEKA Instance object and puts it in memory.
     * @param target
     * @param fetched_data
     */
    public static void readData(ArrayList<ArrayList<String>> target, Instances fetched_data) {
      //Fetch data from the .arff file. Using WEKA's packages for parsing data.
        for(Instance d : fetched_data) {
            String row = d.toString();
            String[] values = row.split(",");
            ArrayList<String> temp = new ArrayList<>();
            for(String s : values) {
                temp.add(s);
            }
            target.add(temp);
        }
    }
    
    
    /**
     * This method returns the passed data with the specified real(continuous) attribute discretized on the basis of the specified split value.
     * @param data the data to discretize
     * @param split the split value
     * @param index the column number of the attribute in the data
     * @return discretized data
     */
    public static ArrayList<ArrayList<String>> discretizeContinuousAtt(ArrayList<ArrayList<String>> data, double split, int index) {
        ArrayList<ArrayList<String>> dummy = new ArrayList<>();
        
        for(ArrayList<String> row : data) {
            //Copy the entire row first.
            ArrayList<String> temp = new ArrayList<>();
            temp.addAll(row);
            
            //Convert the specified continuous attribute into a binary value
            double value = Double.parseDouble(row.get(index));
            if(value <= split)
                value = 0;
            else
                value = 1;
            temp.set(index, Double.toString(value));
            dummy.add(temp);
        }
        return dummy;
    }
    
    
    /**
     * Extracts all the labels from the data.
     * @param data
     * @return list of labels.
     */
    public static ArrayList<String> extractLabels(ArrayList<ArrayList<String>> data) {
        ArrayList<String> labels = new ArrayList<>();
        for(ArrayList<String> row : data) {
            labels.add(row.get(row.size()-1));
        }
        return labels;
    }
    
    
    /**
     * Accessory method to compute the information gain of an attribute -> given as 'i' ; ith attribute in the dataset.
     * The method computes the information gain on the dicretized data after the potential split has been made.
     * @param input_data
     * @param i
     * @param numOfAtt
     * @param scale
     * @return information Gain
     */
    public static double InformationGain(ArrayList<ArrayList<String>> input_data, int i) {
        //Hash to a map rows that have the same value for this attribute.
        Map<String, ArrayList<ArrayList<String>>> map = new HashMap<>();
        for(ArrayList<String> l : input_data) {
            if(map.containsKey(l.get(i))) {
                map.get(l.get(i)).add(l);
            }
            else {
                ArrayList<ArrayList<String>> temp = new ArrayList<>();
                temp.add(l);
                map.put(l.get(i), temp);
            }
        }
        
        //Compute H(Y) for the given input_data data set
        double entropy = 0;
        Map<String, Integer> labels = new HashMap<>();
        for(ArrayList<String> l : input_data) {
            String label = l.get(decisionTree.numOfAtt-1);
            if(labels.containsKey(label)) {
                labels.put(label, labels.get(label)+1);
            }
            else
                labels.put(label, 1);
        }
        
        BigDecimal[] values = new BigDecimal[labels.size()];
        int k = 0;
        for(String l : labels.keySet()) {
            BigDecimal temp = new BigDecimal(labels.get(l));
            values[k] = temp;
            k++;
        }
        entropy = computeEntropy(values).doubleValue();

        double conditionalEntropy = 0;
        for(String s : map.keySet()) {
            ArrayList<ArrayList<String>> list = map.get(s);

            labels = new HashMap<>();
            for(ArrayList<String> l : list) {
                String label = l.get(decisionTree.numOfAtt-1);
                if(labels.containsKey(label)) {
                    labels.put(label, labels.get(label)+1);
                }
                else
                    labels.put(label, 1);
            }
            
            values = new BigDecimal[labels.size()];
            k = 0;
            for(String l : labels.keySet()) {
                BigDecimal temp = new BigDecimal(labels.get(l));
                values[k] = temp;
                k++;
            }
            conditionalEntropy += (list.size()*computeEntropy(values).doubleValue())/((double)input_data.size());
        }
        return entropy - conditionalEntropy;
    }
    
    
    /**
     * Accessory method for calculating Entropy. Need this for the computation of Info Gain.
     * @param values
     * @return entropy of the given values.
     */
    private static BigDecimal computeEntropy(BigDecimal[] values) {
        if(values == null || values.length == 0)
            return new BigDecimal(0);
        
        BigDecimal total = new BigDecimal(0);
        
        for(BigDecimal d : values)
            total = total.add(d);
        
        BigDecimal entropy = new BigDecimal(0);
        
        //Compute the components of the entropy for each value and add to the variable entropy.
        for(BigDecimal value : values) {
            if(value.compareTo(new BigDecimal(0)) == 0)
                continue;
            
            BigDecimal probability = value.divide(total, decisionTree.scale, RoundingMode.HALF_UP);
            double log = Math.log(probability.doubleValue());
            BigDecimal log_prob = new BigDecimal(log/Math.log(2)).setScale(decisionTree.scale, RoundingMode.HALF_UP);
            
            probability = probability.multiply(log_prob);
            entropy = entropy.add(probability);
        }
        return entropy.multiply(new BigDecimal(-1));
    }
    
    /**
     * This method sets the hyperparameters max_depth and info_gain threshold if passed to the program.
     * @param args
     */
    public static void setHyperParameters(String[] args) {
      //Optional argument for max_depth
        if(args.length >= 4) {
            int temp = Integer.parseInt(args[3]);
            if(temp >= 0)
                decisionTree.max_depth = temp;
        }
        
        //Optional argument for information gain threshold
        if(args.length >= 5) {
            double temp = Double.parseDouble(args[4]);
            if(temp >= 0)
                decisionTree.info_Gain_cutoff = temp;
        }
    }
}
