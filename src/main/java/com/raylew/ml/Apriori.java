package com.raylew.ml;

import java.util.*;

/**
 * Created by Raymond on 2017/3/21.
 */
public class Apriori {
    public final static int min_sup=5;

    public Set<Set<String>> findFrequentItemSets(List<Transaction> tList){
        Iterator<Transaction> iterator = tList.iterator();
        Set<Set<String>> set=new HashSet<>();
        while(iterator.hasNext()){
            iterator.next();
        }
        return set;
    }
}
class Transaction{
    private int index;
    private Set<String> set;
}
