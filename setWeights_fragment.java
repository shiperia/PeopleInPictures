package com.siperia.peopleinphotos;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import android.app.Activity;
import android.app.DialogFragment;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.widget.SeekBar;
import android.widget.SeekBar.OnSeekBarChangeListener;


public class setWeights_fragment extends DialogFragment {
	private View view;
	private List<Integer> weights;
	private Listener mListener;
	
	public setWeights_fragment() {super();}
	
	public interface Listener {
		public void sendWeights( List<Integer> w );
	}
	
	public void setListener(Listener listener) {
        mListener = listener;
    }
	
	public void setArgs(List<Integer> weights) {
		if (weights.size() != 5) weights = new ArrayList<Integer>( Arrays.asList(1,1,1,1,1) );
		
		this.weights = weights;
	}
	
	@Override
	public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
	}
	
	@Override
	public void onDestroy() {
		mListener.sendWeights( weights );
		super.onDestroy();
	}
	
	public List<Integer> getWeights() {
		return weights;
	}
			
	//@Override protected void onStop() {super.onStop();}
	
	@Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
            Bundle savedInstanceState) {
		
		view = inflater.inflate(R.layout.similarity_weights_picker, container);
		//getDialog().setTitle(R.string.set_weight_fragment_dialog_title);
		
		getDialog().getWindow().requestFeature(Window.FEATURE_NO_TITLE);
		
		final SeekBar seek1 = (SeekBar)view.findViewById(R.id.seekBar1);
		final SeekBar seek2 = (SeekBar)view.findViewById(R.id.seekBar2);
		final SeekBar seek3 = (SeekBar)view.findViewById(R.id.seekBar3);
		final SeekBar seek4 = (SeekBar)view.findViewById(R.id.seekBar4);
		final SeekBar seek5 = (SeekBar)view.findViewById(R.id.seekBar5);
		
		seek1.setProgress(weights.get(0));
		seek2.setProgress(weights.get(1));
		seek3.setProgress(weights.get(2));
		seek4.setProgress(weights.get(3));
		seek5.setProgress(weights.get(4));
		
		seek1.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override public void onStopTrackingTouch(SeekBar seekBar) { }            	    	   
		    @Override public void onStartTrackingTouch(SeekBar seekBar) { }
			@Override public void onProgressChanged(SeekBar seek1, int progress, boolean fromUser) {
				weights.set(0, progress);
			}
		});
		seek2.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override public void onStopTrackingTouch(SeekBar seekBar) { }            	    	   
		    @Override public void onStartTrackingTouch(SeekBar seekBar) { }
			@Override public void onProgressChanged(SeekBar seek2, int progress, boolean fromUser) {
				weights.set(1, progress);
			}
		});
		seek3.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override public void onStopTrackingTouch(SeekBar seekBar) { }            	    	   
		    @Override public void onStartTrackingTouch(SeekBar seekBar) { }
			@Override public void onProgressChanged(SeekBar seek3, int progress, boolean fromUser) {
				weights.set(2, progress);
			}
		});
		seek4.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override public void onStopTrackingTouch(SeekBar seekBar) { }            	    	   
		    @Override public void onStartTrackingTouch(SeekBar seekBar) { }
			@Override public void onProgressChanged(SeekBar seek4, int progress, boolean fromUser) {
				weights.set(3, progress);
			}
		});
		seek5.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override public void onStopTrackingTouch(SeekBar seekBar) { }            	    	   
		    @Override public void onStartTrackingTouch(SeekBar seekBar) { }
			@Override public void onProgressChanged(SeekBar seek5, int progress, boolean fromUser) {
				weights.set(4, progress);
			}
		});
		
		return view;
	}

}
