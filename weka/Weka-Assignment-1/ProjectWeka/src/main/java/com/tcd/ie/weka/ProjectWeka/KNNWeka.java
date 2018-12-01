package com.tcd.ie.weka.ProjectWeka;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class KNNWeka {
	public static void main(String[] args) throws Exception{
		DataSource source = new DataSource("src/main/java/resources/knn.arff");
		Instances dataset = source.getDataSet();
		dataset.setClassIndex(dataset.numAttributes()-1);
		/**
		 * KNN model
		 */
		IBk ibk = new IBk();
		ibk.buildClassifier(dataset);
		ibk.setKNN(4);
		Evaluation knn = new Evaluation(dataset);
		knn.evaluateModel(ibk, dataset);
		knn.crossValidateModel(ibk, dataset, 10, new Random(1));
		System.out.println(knn.toSummaryString());
	}
}
