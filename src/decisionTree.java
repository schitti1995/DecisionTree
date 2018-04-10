import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

//Download WEKA and include weka.jar in the build path.

public class decisionTree {
    /**
     * Each dataset is an ArrayList of rows. Each row is another ArrayList.
     * originalData holds the training data.
     */
    static ArrayList<ArrayList<String>> originalData = new ArrayList<>();
    
    /**
     * attributes holds the list of attributes of the data in the training dataset.
     */
    static List<Attribute> attributes = new ArrayList<>();
    
    
    /**
     * Number of columns(# of attributes+label) in the dataset.
     */
    static int numOfAtt = 0;
    
    /**
     * Scale of the floating point arithmetic.
     */
    static final int scale = 10;
    
    
    /**
     * Hyperparameters for training the tree.
     */
    static int max_depth = 4;
    static double info_Gain_cutoff = 0.1;
    
    
    /**
     * Needs runtime arguments (At least 3, at most 5).
     * args[0] --> Path to the training data file.
     * args[1] --> Path to the test data file.
     * args[2] --> Path to the output file to write the test labels to.
     * args[3] --> Integer specifying the max_depth of the tree. (optional)
     * args[4] --> Double value specifying the information gain cut-off (optional)
     * Sample command line arguments: ./src/trainProdSelection.arff ./src/testProdSelection.arff ./src/output.txt 4 0.1
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        
        //Set the hyperparameters if they are passed at run time.
        AuxMethods.setHyperParameters(args);
        
        //Read training data into memory.
        BufferedReader reader = new BufferedReader(new FileReader(args[0]));
        ArffReader arff = new ArffReader(reader);
        Instances fetched_data = arff.getData();
        AuxMethods.readData(originalData, fetched_data);
        
        System.out.println("Training data read...\n");
        
        //Fetch the Attributes in the data
        for(int i = 0; i < fetched_data.numAttributes(); i++) {
            attributes.add(fetched_data.attribute(i));
        }
        numOfAtt = attributes.size();
        
        
        //Cross-validate to find the optimal height of the decision tree.
        //After this method is done executing, it returns the average validation error for this max_depth.
        System.out.println("Cross-validating to find the optimal hyperparameters...\n");
        CrossValidation.crossValidate(originalData);
        
        
        //Now that hyperparameter optimisation is done, build the decision tree,
        //make predictions and compute error on the whole training dataset.
        TreeNode root = new TreeNode();
        trainTree(originalData, root, 0);
        System.out.println("Training data accuracy = " + predict(root, originalData));
        
        
        //Print the decision tree.
        System.out.println();
        System.out.println("Printing the tree......");
        AuxMethods.printTree(root);
        
        reader.close();
        
        //COMMENT ALL CODE FROM THIS POINT ON IF MAKING PREDICTIONS ON A TEST FILE IS NOT NECESSARY.
        
        //Make predictions on the test dataset if it has been passed.
        BufferedReader reader1 = new BufferedReader(new FileReader(args[1]));
        ArffReader arff1 = new ArffReader(reader1);
        fetched_data = arff1.getData();
        ArrayList<ArrayList<String>> test = new ArrayList<>();
        AuxMethods.readData(test, fetched_data);
        
        System.out.println("\nTest data read...");
        System.out.println("Predicting...");
        
        predict(root, test, args[2]);
        System.out.println("Predictions on test data written to a file. Bye bye!");
        reader.close();
    }


    /**
     * This variant of predict() makes predictions on the given labeled dataset and reports the accuracy of the
     * predicted labels by comparing with the actual labels.
     * @param root
     * @param data
     * @return accuracy of the labels.
     */
    static double predict(TreeNode root, ArrayList<ArrayList<String>> data) {
        List<String> actual = AuxMethods.extractLabels(data);
        List<String> predicted = new ArrayList<>();
        
        for(ArrayList<String> row : data) {
            predicted.add(predictRow(root, row));
        }
        
        //Compute error on the data.
        int size = actual.size();
        double error = 0;
        
        for(int i = 0; i < size; i++) {
            if(!actual.get(i).equals(predicted.get(i)))
                error++;
        }
        
        //System.out.println("Error = " + (error/size * 100));
        return 100 - ((error/size)*100);
    }
    
    
    /**
     * This variant of predict() predicts the labels on the test dataset and writes them to a file. 
     * @param root
     * @param test test data file
     * @param string path to the output file
     */
    private static void predict(TreeNode root, ArrayList<ArrayList<String>> test, String outputFile) {
        List<String> predicted = new ArrayList<>();
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(outputFile));
            for(ArrayList<String> row : test) {
                predicted.add(predictRow(root, row));
            }
        
            for(String label : predicted) {
                bw.write(label + "\n");
            }
            
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    
    /**
     * Returns the prediction made on a record by the given decision tree.
     * @param root  the built decision tree
     * @param row   the record to make predictions on
     * @return
     */
    private static String predictRow(TreeNode root, ArrayList<String> row) {
        //If you reached a leaf, print its label.
        if(root.isLeaf) {
            //System.out.println(root.label);
            return root.label;
        }
        
        String att_type = root.attribute.type;
        if(att_type.equals("nominal")) {
            for(TreeNode child : root.adj) {
                if(child.prev_SplitVal.equals(row.get(root.attribute.m))) {
                    return predictRow(child, row);
                }
            }
        }
        else {
            double value = root.attribute.split;
            int att = root.attribute.m;
            if(Double.parseDouble(row.get(att)) <= value) {
                return predictRow(root.adj.get(0), row);
            }
            else
                return predictRow(root.adj.get(1), row);
        }
        return null;
    }
    
    
    /**
     * This method builds the decision tree.
     * The stopping criteria are the hyperparameters 1) max_depth, 2) information gain threshold. 
     * @param data
     * @param root
     * @param current_depth
     */
    static void trainTree(ArrayList<ArrayList<String>> data, TreeNode root, int current_depth) {
        BestAttribute best = findBestAttribute(data);
        
        if(AuxMethods.InformationGain(best.data, best.m) <= info_Gain_cutoff) {
            root.isLeaf = true;
            root.label = majorityClassifier(data);
            return;
        }
        
        //Check if the node has reached the max specified depth.
        if(current_depth == max_depth) {
            root.isLeaf = true;
            root.label = majorityClassifier(data);
            return;
        }

        List<String> list = getPossibleValues(data, best);
        
        if(list.size() == 1) {
            root.isLeaf = true;
            root.label = majorityClassifier(data);
            return;
        }
        
        //Create a node for each value of the split attribute
        if(best.type.equals("nominal")) {
            for(String s : list) {
                TreeNode child = new TreeNode();
                child.prev_SplitVal = s;
                root.adj.add(child);
            }
        }
        
        //If numeric, there are only 2 children - 1) <=split value, 2) >split value
        else if(best.type.equals("numeric")) {
            TreeNode left = new TreeNode();
            left.prev_SplitVal = "<" + Double.toString(best.split);
            root.adj.add(left);
            
            TreeNode right = new TreeNode();
            right.prev_SplitVal = ">" + Double.toString(best.split);
            root.adj.add(right);
        }
        
        root.attribute = best;
        
        //recurse on the node's children
        for(TreeNode next : root.adj) {
            ArrayList<ArrayList<String>> reduced_dataset1 = ReduceDataSet(data, best, next.prev_SplitVal);
            trainTree(reduced_dataset1, next, current_depth+1);
        }
    }
    
    
    /**
     * This method is the majority vote classifier used by the decision tree to get the labels at the leaves.
     * @param input_data
     * @return the label that occurs the most number of times.
     */
    private static String majorityClassifier(ArrayList<ArrayList<String>> input_data) {
        Map<String, Integer> map = new HashMap<>();
        for(ArrayList<String> row : input_data) {
            String label = row.get(numOfAtt-1);
            if(map.containsKey(label))
                map.put(label, map.get(label)+1);
            else
                map.put(label, 1);
        }
        
        int max = -1;
        String toReturn = null;
        for(String s : map.keySet()) {
            if(max < map.get(s)) {
                max = map.get(s);
                toReturn = s;
            }
        }
        
        return toReturn;
    }
    
    
   /**
    * Return a list of all possible values of a given attibute in the given data set.
    * @param input_data
    * @param best
    * @return list of lists
    */
    private static List<String> getPossibleValues(ArrayList<ArrayList<String>> input_data, BestAttribute best) {
        Set<String> set = new HashSet<>();
        if(best.type.equals("nominal")) {
            for(ArrayList<String> row : input_data) {
                String val = row.get(best.m);
                if(!set.contains(val))
                    set.add(val);
            }
        }
        else {
            for(ArrayList<String> row : input_data) {
                String val = row.get(best.m);
                if(Double.parseDouble(val) <= best.split)
                    set.add("0");
                else
                    set.add("1");
            }
        }
        List<String> list = new ArrayList<>();
        for(String s : set) {
            list.add(s);
        }
        
        return list;
    }
    
    
    /**
     * Input: Current data set, splitting attribute, attribute value.
     * Output: Return a subset of the given dataset that contains only those records for which the specified attribute has the specified value.
     * @param input_data
     * @param best
     * @param prev_SplitVal
     * @return list of lists
     */
    private static ArrayList<ArrayList<String>> ReduceDataSet(ArrayList<ArrayList<String>> input_data, BestAttribute best, String prev_SplitVal) {
        ArrayList<ArrayList<String>> toReturn = new ArrayList<>();
        if(best.type.equals("nominal")) {
            for(ArrayList<String> row : input_data) {
                if(row.get(best.m).equals(prev_SplitVal)) {
                    toReturn.add(row);
                }
            }
        }
        else {
            char inequality = prev_SplitVal.charAt(0);
            for(ArrayList<String> row : input_data) {
                double val = Double.parseDouble(row.get(best.m));
                if(inequality == '<') {
                    if(val <= best.split) {
                        toReturn.add(row);
                    }
                }
                else if(inequality == '>') {
                    if(val > best.split) {
                        toReturn.add(row);
                    }
                }
            }
        }
        return toReturn;
    }
    
    
    /**
     * Find the attribute that "best" splits the given data (Criterion for "best" --> Information Gain).
     * @param data
     * @return the best attriute object.
     */
    private static BestAttribute findBestAttribute(ArrayList<ArrayList<String>> data) {
        BestAttribute best = null;
        double best_infoGain = Double.MIN_VALUE;
        
        //Traverse through all attributes and find the "best" splitting node on the given data.
        for(int i = 0; i < attributes.size()-1; i++) {
            Attribute current = attributes.get(i);
            //System.out.println(current);
            String type = Attribute.typeToString(current);
            
            if(type.equals("nominal")) {
                double inf_Gain = AuxMethods.InformationGain(data, i);
                if(i == 0) {
                    best_infoGain = inf_Gain;
                    best = new BestAttribute("nominal", 0, i, data);
                }
                else {
                    if(best_infoGain < inf_Gain) {
                        best_infoGain = inf_Gain;
                        best = new BestAttribute("nominal", 0, i, data);
                    }
                }
            }
            else if(type.equals("numeric")) {
                for(ArrayList<String> row : data) {
                    double split = Double.parseDouble(row.get(i));
                    ArrayList<ArrayList<String>> dummy = AuxMethods.discretizeContinuousAtt(data, split, i);
                    double inf_Gain = AuxMethods.InformationGain(dummy, i);
                    if(i == 0) {
                        best_infoGain = inf_Gain;
                        best = new BestAttribute("numeric", split, i, dummy);
                    }
                    else {
                        if(best_infoGain < inf_Gain) {
                            best_infoGain = inf_Gain;
                            best = new BestAttribute("numeric", split, i, dummy);
                        }
                    }
                }
            }
            else {
                //Doesn't make sense for this case to ever occur.
                System.out.println("Neither nominal nor numeric! :( ");
            }
        }
        return best;
    }
}
