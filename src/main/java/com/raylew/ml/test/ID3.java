package com.raylew.ml.test;

/**
 * Created by Raymond on 2017/3/14.
 */


import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;


class Instance {
    /**
     * DO NOT CHANGE this class
     */
    static int FTSVALUERANGE = 4; // fts can only be 0,1,2,3
    int label; // label=1 ==> +; label=0 ==> -; label=-1 ==> cannot decide
    int[] fts; // mapping a t c g TO 0,1,2,3. For example fts[3]=2 means 3rd
    // feature is 'c'
    int uniqueId;// every instance will have an uniqe Id;

    public Instance(String line, int id) {
        this.uniqueId = id;
        String[] temp = line.split(" ");
        // set this.label
        if (temp[1].equals("+")) {
            label = 1;
        } else {
            label = 0;
        }
        // set this.fts
        char[] charfts = temp[0].toCharArray();
        this.fts = new int[charfts.length];
        for (int i = 0; i < charfts.length; i++) {
            if (charfts[i] == 'a') {
                fts[i] = 0;
            } else if (charfts[i] == 't') {
                fts[i] = 1;
            } else if (charfts[i] == 'c') {
                fts[i] = 2;
            } else {
                fts[i] = 3;
            }
        }
    }
}

class Node {

    Node parent;
    Node children[];

    /**
     * The test feature for internal node
     */
    int testFts;

    int numOfFts;

    List<Instance> instances;

    int predictedLabel = -1;

    Node(Node parent, List<Instance> instances) {
        this.parent = parent;
        children = new Node[Instance.FTSVALUERANGE];

        this.instances = instances;
        numOfFts = instances.get(0).fts.length;
        testFts = -1;

        int count[] = {0, 0};
        for (Instance t : this.instances)
            count[t.label]++;
        predictedLabel = count[0] > count[1] ? 0 : 1;
    }

    public int classify(Instance t) {
        // System.out.println("test\t" + testFts);
        if (testFts == -1) {
            return predictedLabel;
        } else {
            if (children[t.fts[testFts]] != null) {
                return children[t.fts[testFts]].classify(t);
            } else {
                return -1;// cannot decide; return parent label
            }
        }
    }

    /**
     * Exports the decision tree rooted from this node to a TreeML file.
     *
     * @param filename The name of the target file
     */
    public void writeTreeXML(String filename) {
        try {
            PrintWriter writer = new PrintWriter(new FileWriter(filename));
            writer.println("<?xml version=\"1.0\" ?>");
            writer.println("<tree>");
            writer.println("<declarations>");
            writer.println("<attributeDecl name=\"name\" type=\"String\" />");
            writer.println("<attributeDecl name=\"weight\" type=\"Real\" />");
            writer.println("</declarations>");
            writeTreeML(writer);
            writer.println("</tree>");
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void writeTreeML(PrintWriter writer) {
        if (testFts != -1)
            writer.println("<branch>");
        else
            writer.println("<leaf>");

        writer.print("<attribute name=\"name\" value=\"");
        if (parent == null)
            writer.print("root");
        else {
            for (int i = 0; i < Instance.FTSVALUERANGE; i++) {
                if (this == parent.children[i]) {
                    writer.print("fts" + parent.testFts + " = " + i);
                }
            }
        }
        writer.println("\" />");

        if (testFts != -1) {
            for (int i = 0; i < Instance.FTSVALUERANGE; i++) {
                if (children[i] != null)
                    children[i].writeTreeML(writer);
            }
            writer.println("</branch>");
        } else {
            writer.println("<attribute name=\"weight\" value=\""
                    + instances.size() + "\" />");
            writer.println("</leaf>");
        }
    }

}

public class ID3 {
    Node root;

    /**
     * @author Congle Zhang
     */
    public abstract static class ImpurityFunction {
        public abstract double calc(int a, int b);
    }

    /**
     * Impurity function using entropy.
     */
    public static ImpurityFunction impurity_entropy = new ImpurityFunction() {
        public double calc(int a, int b) {
            double pa = a / ((double) a + (double) b);
            double pb = b / ((double) a + (double) b);

            double res = 0;
            if (a > 0)
                res += -pa * Math.log(pa);
            if (b > 0)
                res += -pb * Math.log(pb);

            return res / Math.log(2);
        }
    };

    /**
     * Impurity function using misclassification.
     */
    public static ImpurityFunction impurity_misclassification = new ImpurityFunction() {
        public double calc(int a, int b) {
            if (a > b) {
                return b / (double) (a + b);
            }
            return a / (double) (a + b);
        }
    };

    /**
     * Generates the decision tree with the given impurity function and
     * chi-square test.
     *
     * @param instances The training data.
     * @param f      The impurity function.
     * @return The trained decision tree.
     */
    public static Node generate(List<Instance> instances, ImpurityFunction f) {
        Node root = new Node(null, instances);
        expand(root, f, ID3.chi_square_100, 0);// when use ID3.chi_square_100,
        // there is no prunning,
        return root;
    }

    public static Node generate(List<Instance> instances, ImpurityFunction f,
                                ChiSquareTest cst) {
        Node root = new Node(null, instances);
        expand(root, f, cst, 0);
        return root;
    }

    static void expand(Node node, ImpurityFunction impurityFunction,
                       ChiSquareTest cst, int depth) {

        double maxGain = -100000;
        int maxGainDecision = -1;
        int num = node.instances.size();
        int ftsNum = node.instances.get(0).fts.length;
        int mcount[][] = new int[Instance.FTSVALUERANGE][2];

        int parentPos = 0, parentNeg = 0;
        for (int i = 0; i < node.instances.size(); i++) {
            if (node.instances.get(i).label == 1) {
                parentPos++;
            } else {
                parentNeg++;
            }
        }
        /* Iterate over all attributes, find the best attribute */
        for (int s = 0; s < node.numOfFts; ++s) {

            int count[][] = new int[Instance.FTSVALUERANGE][2];
            for (Instance t : node.instances) {
                if (t.label == 1)
                    count[t.fts[s]][1]++;
                else
                    count[t.fts[s]][0]++;
            }
            double gain = impurityFunction.calc(parentPos, parentNeg);
            for (int i = 0; i < Instance.FTSVALUERANGE; i++) {
                gain -= 1.0 * (count[i][0] + count[i][1])
                        / (parentPos + parentNeg)
                        * impurityFunction.calc(count[i][0], count[i][1]);
                // error += Math.min(count[i][0], count[i][1]);
            }

            if (gain > maxGain) {
                maxGain = gain;
                maxGainDecision = s;
                for (int i = 0; i < Instance.FTSVALUERANGE; i++) {
                    mcount[i][0] = count[i][0];
                    mcount[i][1] = count[i][1];
                }
            }

        }

        /** If maxGain ==0 then the node is pure, stop growing the tree */
        if (maxGain > 1e-10 && cst.test(mcount)) {
            node.testFts = maxGainDecision;

            ArrayList<ArrayList<Instance>> ts = new ArrayList<ArrayList<Instance>>();
            for (int i = 0; i < Instance.FTSVALUERANGE; ++i) {
                ts.add(new ArrayList<Instance>());
            }

            for (Instance t : node.instances)
                ts.get(t.fts[maxGainDecision]).add(t);

			/* Grow the tree recursively */
            for (int i = 0; i < Instance.FTSVALUERANGE; i++) {
                if (maxGainDecision == 16 && i == 2) {
                    int x = 0;
                }
                if (ts.get(i).size() > 0) {
                    node.children[i] = new Node(node, ts.get(i));
                    expand(node.children[i], impurityFunction, cst, depth + 1);
                }
            }
        }
    }

    public void learn(ArrayList<Instance> instances, ImpurityFunction f,
                      ChiSquareTest cts) {
        this.root = generate(instances, f, cts);
    }

    public void learn(ArrayList<Instance> instances, ImpurityFunction f) {
        this.root = generate(instances, f);
    }

    public void learn(List<Instance> instances) {
        this.root = generate(instances, ID3.impurity_entropy);
    }

    public static class ChiSquareTest {
        double threshold;

        /**
         * Creates a chi-square test with the given threshold.
         *
         * @param threshold The value of the threshold.
         */
        ChiSquareTest(double threshold) {
            this.threshold = threshold;
        }

        /**
         * Calculate the chi-square statistic.
         *
         * @param count The splitting result.
         * @return true if the chi-square statistic is greater than the
         * threshold.
         */
        public boolean test_old(int[][] count) {
            double chi_square = 0;
            double num = count[0][0] + count[0][1] + count[1][0] + count[1][1];
            int j = 0;

            for (int i = 0; i < 2; ++i) {
                double ne = (count[j][0] + count[j][1]) / num
                        * (count[0][i] + count[1][i]);
                double nl = count[j][i];
                chi_square += (nl - ne) * (nl - ne) / ne;
            }

            return chi_square > threshold + 1e-8;
        }

        public boolean test(int[][] count) {
            double chi_square = 0;
            int m_x_l = count.length;
            int m_y_l = count[0].length;
            double m_x[] = new double[m_x_l];// margin for x
            double m_y[] = new double[m_y_l];// margin for y
            double m = 0;
            for (int i = 0; i < m_x_l; i++) {
                for (int j = 0; j < m_y_l; j++) {
                    m_x[i] += count[i][j];
                    m += count[i][j];
                }
            }
            for (int j = 0; j < m_y_l; j++) {
                for (int i = 0; i < m_x_l; i++) {
                    m_y[j] += count[i][j];
                }
            }
            for (int i = 0; i < m_x_l; i++) {
                for (int j = 0; j < m_y_l; j++) {
                    double e_ij = 1.0 * m_x[i] * m_y[j] / m;
                    double o_ij = count[i][j];
                    if (e_ij > 0) {
                        chi_square += (e_ij - o_ij) * (e_ij - o_ij) / e_ij;
                    }
                }
            }
            return chi_square > threshold + 1e-8;
        }
    }

    /**
     * Chi-square test with threshold .001.
     */
    public static ChiSquareTest chi_square_001 = new ChiSquareTest(16.27);
    /**
     * Chi-square test with threshold .01.
     */
    public static ChiSquareTest chi_square_01 = new ChiSquareTest(11.34);
    /**
     * Chi-square test with threshold .05.
     */
    public static ChiSquareTest chi_square_05 = new ChiSquareTest(7.82);
    /**
     * Chi-square test with threshold 1.
     */
    public static ChiSquareTest chi_square_100 = new ChiSquareTest(0);

    public List<Integer> classify(List<Instance> testInstances) {
        List<Integer> predictions = new ArrayList<Integer>();
        for (Instance t : testInstances) {
            // System.out.println("instance" + t.uniqueId);
            int predictedCategory = root.classify(t);
            predictions.add(predictedCategory);
        }
        return predictions;
    }

    public static void load(String trainfile, String testfile,
                            List<Instance> trainInstances, List<Instance> testInstances) {
        int UNIQEID = 0;
        try {
            ClassLoader classLoader = ID3.class.getClassLoader();
            BufferedReader br = new BufferedReader(new FileReader(classLoader.getResource(trainfile).getFile()));
            String line;
            while ((line = br.readLine()) != null) {
                Instance ins = new Instance(line, UNIQEID++);
                trainInstances.add(ins);
            }
            br.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        try {
            ClassLoader classLoader = ID3.class.getClassLoader();
            BufferedReader br = new BufferedReader(new FileReader(classLoader.getResource(testfile).getFile()));
            String line;
            while ((line = br.readLine()) != null) {
                Instance ins = new Instance(line, UNIQEID++);
                testInstances.add(ins);
            }
            br.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static double computeAccuracy(List<Integer> predictions,
                                         List<Instance> testInstances) {
        if (predictions.size() != testInstances.size()) {
            return 0;
        } else {
            int right = 0, wrong = 0;
            for (int i = 0; i < predictions.size(); i++) {
                if (predictions.get(i) == null) {
                    wrong++;
                } else if (predictions.get(i) == testInstances.get(i).label) {
                    right++;
                } else {
                    wrong++;
                }
            }
            return right * 1.0 / (right + wrong);
        }
    }

    /**
     * Usage:
     * javac ID3
     * java ID3
     */
    public static void main(String[] args) {
        ArrayList<Instance> trainInstances = new ArrayList<Instance>();
        ArrayList<Instance> testInstances = new ArrayList<Instance>();
        load("training.txt", "test.txt", trainInstances,
                testInstances);
        {
            ID3 id3 = new ID3();
            id3.learn(trainInstances);//by default ID3: no pruning; information gain;
            id3.root.writeTreeXML("tree_fulltree.xml");
            List<Integer> trainpredictions = id3.classify(trainInstances);
            System.out.println("ID3 with full tree on training\t"
                    + id3.computeAccuracy(trainpredictions, trainInstances));

            List<Integer> predictions = id3.classify(testInstances);
            System.out.println("ID3 with full tree on test\t"
                    + id3.computeAccuracy(predictions, testInstances));
        }
    }

}
