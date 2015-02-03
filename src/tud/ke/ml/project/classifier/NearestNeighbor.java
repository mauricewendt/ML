package tud.ke.ml.project.classifier;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import tud.ke.ml.project.framework.classifier.ANearestNeighbor;
import tud.ke.ml.project.util.Pair;
import weka.core.Attribute;

/**
 * This implementation assumes the class attribute is always available (but probably not set)
 * @author cwirth
 *
 */
public class NearestNeighbor extends ANearestNeighbor {
	
	protected double[] scaling;
	protected double[] translation;
	
	private List<List<Object>> traindata;
	
	@Override
	protected Object vote(List<Pair<List<Object>, Double>> subset) {
		Map<Object, Double> votes = isInverseWeighting() ? getWeightedVotes(subset) : getUnweightedVotes(subset);
		return getWinner(votes);
	}
	@Override
	protected void learnModel(List<List<Object>> traindata) {
		this.traindata = traindata;
		
	}
	@Override
	protected Map<Object, Double> getUnweightedVotes(
			List<Pair<List<Object>, Double>> subset) {
		
		Map<Object, Double> votes = new HashMap<>();
		Object classAttribute = null;
		
		for (Pair<List<Object>, Double> pair : subset) {
			if(pair==null)
				continue;
			
			classAttribute = pair.getA().get(getClassAttribute());
			if(votes.containsKey(classAttribute))
				votes.put(classAttribute, votes.get(classAttribute)+1.0);
			else
				votes.put(classAttribute, 1.0);

		}
		
		return votes;
	}
	@Override
	protected Map<Object, Double> getWeightedVotes(
			List<Pair<List<Object>, Double>> subset) {
		
		Map<Object, Double> votes = new HashMap<>();
		Object key = null;
		
		for (Pair<List<Object>, Double> pair : subset) {
			if(pair==null)
				continue;
			key = pair.getA().get(getClassAttribute());
			if(votes.containsKey(key))
				votes.put(key, votes.get(key)+(1/Math.pow(pair.getB(), 2)));
			else
				votes.put(key, 1/Math.pow(pair.getB(), 2));
			
		}
		
		return votes;
	}
	@Override
	protected Object getWinner(Map<Object, Double> votesFor) {
		Object winner = null;
		Double highestVote = 0.0;
		for (Map.Entry<Object,Double> entry : votesFor.entrySet()) {
			if(entry.getValue()>highestVote){
				winner = entry.getKey();
				highestVote = entry.getValue();
			}
		}
		return winner;
	}
	@Override
	protected List<Pair<List<Object>, Double>> getNearest(List<Object> testdata) {
		List<Pair<List<Object>,Double>> kNearestNeighbours = new ArrayList<>(getkNearest());
		
		for (int i = 0; i < getkNearest(); i++) {
			kNearestNeighbours.add(null);
		}

		double newWeight;
		
		Pair<List<Object>,Double> currentNeighbour = null;
		int index = -1;
		boolean newKNearest = false;
		
		Pair<List<Object>, Double> newItem = null;
		
		for (List<Object> compareObj : traindata) {
			newWeight = (getMetric()==0) ? determineManhattanDistance(testdata, compareObj) : determineEuclideanDistance(testdata, compareObj);
			
			for (int i = 0; i < kNearestNeighbours.size(); i++) {
				currentNeighbour = kNearestNeighbours.get(i);
				if(currentNeighbour == null || currentNeighbour.getB()>newWeight){
					index = i;
					newKNearest = true;
					break;
				}
			}
			
			if(newKNearest){	
				newItem = new Pair<List<Object>, Double>(compareObj, newWeight);
				kNearestNeighbours.remove(index);
				kNearestNeighbours.add(index, new Pair<List<Object>, Double>(compareObj, newWeight));
				newKNearest = false;
			}
			
		}
	
		return kNearestNeighbours;
	}
	@Override
	protected double determineManhattanDistance(List<Object> instance1,
			List<Object> instance2) {
		
		double distance = 0;
		double increase = 0;
		
		Object attr1 = null;
		Object attr2 = null;
		
		for (int i = 0; i < instance1.size(); i++) {
			
			attr1 = instance1.get(i);
			attr2 = instance2.get(i);
			
			
			if(!attr1.equals(attr2)){
				if (attr1 instanceof String && attr2 instanceof String){
					increase = 1;
				}
				else {
					increase = Math.abs((Double) attr1 - (Double) attr2);
				}
				distance+=increase;
			}
		
		}
		return distance;
	}
	
	public boolean isNumeric(Object obj){
		return (obj instanceof Integer || obj instanceof Double);
	}
	@Override
	protected double determineEuclideanDistance(List<Object> instance1,
			List<Object> instance2) {
		
		double distance = 0;
		double increase = 0;
		
		Object attr1 = null;
		Object attr2 = null;
		
		for (int i = 0; i < instance1.size(); i++) {
			
			attr1 = instance1.get(i);
			attr2 = instance2.get(i);
			
			if(!attr1.equals(attr2)){
				
				if (attr1 instanceof String && attr2 instanceof String){
					increase = 1;
				}
				else {
					increase = Math.abs((Double) attr1 - (Double) attr2);
				}
			
				distance+=Math.pow(increase,2);
			}
			
		}
		return Math.sqrt(distance);
	}
	@Override
	protected double[][] normalizationScaling() {
		double[] scale = null;
		double[] transl = null;
		return isNormalizing() ? new double[][]{scale,transl} : null;
	}
	@Override
	protected String[] getMatrikelNumbers() {
		return new String[]{"1802184", "1912896"};
	}

}
