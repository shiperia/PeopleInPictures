package com.siperia.peopleinphotos;

import java.util.*;

/**
 *    Implementation of http://en.wikipedia.org/wiki/Permutation#Systematic_generation_of_all_permutations                             &nbsp;
      Algorithm to efficiently generate permutations of a sequence                                                                                                                   
      until all possibilities are exhausted
      
      Taken and modified from http://42bits.wordpress.com/2010/04/12/generating_all_possible_permutations_of_a_sequence/
         -jn
 */
public class Permute<E>
{
	private int[] arrIdxs;
	private ArrayList<E> arr;
	ArrayList<E> ret = new ArrayList<E>();
		
	public ArrayList<E> get_next()
	{
		ret.clear();
		for (int idx = 0 ; idx < arrIdxs.length;idx++)
			ret.add(idx,arr.get(arrIdxs[idx]));
		//Permute integer based array indexes, which can be used to get permuted array in return
		return ret;
	}
	
	public Permute(ArrayList<E> arr)
	{
		this.arr = new ArrayList<E>();
		for (E each:arr) this.arr.add(each);
		arrIdxs = new int[arr.size()];
		for(int i = 0 ; i < arr.size();i++)    //Set indexes lexicographically minimal value;
			this.arrIdxs[i]= i;
	}

	public boolean next_permutation()
	{
		int i,j,l;

		for(j =arr.size() -2 ;j >=0  ; j--) //get maximum index j for which arr[j+1] > arr[j]
			if (arrIdxs[j+1] > arrIdxs[j])
				break;
		if (j == -1) //has reached it's lexicographic maximum value, No more permutations left 
			return false;

		for(l = arr.size()-1;l > j ; l--) //get maximum index l for which arr[l] > arr[j]
			if (arrIdxs[l] > arrIdxs[j])
				break;

		int swap = arrIdxs[j]; //Swap arr[i],arr[j] 
		arrIdxs[j] = arrIdxs[l];
		arrIdxs[l] = swap;

		for (i = j+1;i < arrIdxs.length;i++) //reverse array present after index : j+1 
		{
			if (i > arrIdxs.length - i + j )
				break;
			swap = arrIdxs[i];
			arrIdxs[i] = arrIdxs[arrIdxs.length - i + j];
			arrIdxs[arrIdxs.length - i + j] = swap;
		}
		return true;
	}
}
