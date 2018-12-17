package com.tcd.ie.weka.ProjectWeka;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class KNNWeka {
	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource("src/main/java/resources/knn.arff");
		Instances dataset = source.getDataSet();
		dataset.setClassIndex(dataset.numAttributes() - 1);
		NumericToNominal convert = new NumericToNominal();
		String[] options = new String[2];
		options[0] = "-R";
		options[1] = "1-2"; // range of variables to make numeric

		convert.setOptions(options);
		convert.setInputFormat(dataset);

		Instances newData = Filter.useFilter(dataset, convert);
		/**
		 * KNN model
		 */
		IBk ibk = new IBk();
		ibk.buildClassifier(newData);
		ibk.setKNN(4);
		Evaluation knn = new Evaluation(newData);
		knn.evaluateModel(ibk, newData);
		for (int i = 0; i <= 10; i++) {
			knn.crossValidateModel(ibk, newData, 10, new Random(i));
			System.out.println(knn.toSummaryString());
		}
	}
}
