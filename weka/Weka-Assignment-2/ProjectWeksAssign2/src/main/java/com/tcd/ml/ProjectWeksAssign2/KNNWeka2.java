package com.tcd.ml.ProjectWeksAssign2;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class KNNWeka2 {
	public static void main(String[] args) throws Exception{
		DataSource source = new DataSource("src/main/java/resources/knn.arff");
		Instances dataset = source.getDataSet();
		dataset.setClassIndex(dataset.numAttributes()-1);
		/**
		 * KNN model
		 */
		IBk ibk = new IBk();
		ibk.setKNN(4);
		ibk.buildClassifier(dataset);
		Evaluation knn = new Evaluation(dataset);
		knn.evaluateModel(ibk, dataset);
		knn.crossValidateModel(ibk, dataset, 10, new Random(1));
		System.out.println(knn.toSummaryString());
	}
}
