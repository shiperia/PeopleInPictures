package com.siperia.peopleinphotos;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.Random;
import java.util.Scanner;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Range;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.CvKNearest;

import com.siperia.peopleinphotos.setWeights_fragment.Listener;

import delaunay_triangulation.Point_dt;
import android.os.Bundle;
import android.os.Environment;
import android.app.AlertDialog;
import android.app.DialogFragment;
import android.app.Fragment;
import android.app.FragmentTransaction;
import android.content.DialogInterface;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;
import android.util.Pair;

import android.view.LayoutInflater;
import android.view.View;
import android.view.View.OnClickListener;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

public class GalleryScanner extends DialogFragment implements Listener {
	
	private static final String		TAG = "PeopleInPhotos::GalleryScanner";
	private static final File		sdDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
	private static final File		identRootDir = new File(sdDir, "PiP_idents");
	private static final String		photoInfoFile = identRootDir.getAbsolutePath() + "/DCIMdata.txt";
	private static final String		matchesFile = identRootDir.getAbsolutePath() + "/matches.txt";
	private static final String		kNNfile = identRootDir.getAbsolutePath() + "/SimilarityKNN.xml";
	private static final String		delim = " "; // delimiter used in photoInfoFile
		
	private View					view;
	private Paint 					paint;
	private Bitmap					bmap;
	private Canvas					canvas;
	
	private int						currentPhoto = -1;
	private String					picturePath;
	private Map<String,List<Similarity>> similarPhotos = new HashMap<String,List<Similarity>>();
													// list of best matches and their scores
													// First index is for file, second (0-9) are 10 best matches
	private List<Integer>			similarity_weights = new ArrayList<Integer>();
	private List<Integer>			before = new ArrayList<Integer>();
	
	private ArrayList<Pair<String,  ArrayList<double[]>>> groups = null;
	private ArrayList<Integer>  	selectedIdx = new ArrayList<Integer>();
	private CvKNearest				similarity_knn = null;
	private static final int 		GROUP_FAMILY = 1, GROUP_GROUP = 2, GROUP_WEDDING = 3;
	private static final int		GROUP_CLASSES = 3;
	
	private ImageView				img;
			
									// How many faces can there be in one photo for graph matching
	private Photo					thisPhoto = null;
	private File[]					files;
	
    private static FaceDetectionAndProcessing faceclass		= null;
    
    static CameraBridgeViewBase		mCV = null;
    
    GalleryScanner 					this_class = null;
    
    public GalleryScanner() {
		picturePath = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM).getAbsolutePath();
		picturePath = picturePath + "/Camera";
		
		this_class = this;
		
		paint = new Paint();
        paint.setColor(Color.GREEN);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(2);
        paint.setTextSize(42);
        
        File dir = new File(picturePath);
		if (dir.isDirectory())
		{
			files = faceclass.getJPGList(dir);
			Log.d(TAG, files.length+" photos in "+picturePath);
		}
				
		loadSimilarities();
		
	}
    
    static GalleryScanner newInstance(FaceDetectionAndProcessing fc, CameraBridgeViewBase mcv) {
        faceclass = fc;
        mCV = mcv;
		
		GalleryScanner f = new GalleryScanner();
        return f;
    }
        
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setStyle(DialogFragment.STYLE_NO_TITLE, android.R.style.Theme_Holo_NoActionBar_Fullscreen);
        mCV.disableView();
    }
    
    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
            Bundle savedInstanceState) {
    	
        view = inflater.inflate(R.layout.activity_gallery_scanner, container);
        img = (ImageView)view.findViewById(R.id.imageView);
        
        Button nextButton = (Button)view.findViewById(R.id.Button_NEXT);
		nextButton.setOnClickListener(new OnClickListener() {
			@Override public void onClick(View v) {
				currentPhoto++;
				if (currentPhoto >= files.length) currentPhoto = 0;
				
				thisPhoto = new Photo( files[currentPhoto].getAbsolutePath(), faceclass, true, false );
				
				bmap = thisPhoto.load();
				img.setImageBitmap(bmap);
												
		    	// Draw the hits on the original picture
				canvas = new Canvas(bmap);
		        canvas.drawText(thisPhoto.faces.size()+" faces", 10, 40, paint);
		        		        
		        for (int i=0; i < thisPhoto.faces.size(); i++) {
		        	Face f = thisPhoto.faces.get(i);
		        	Rect r = f.location;
		        	String tag = "";
		        	
	        		if (faceclass.mIdentifierMethod != FaceDetectionAndProcessing.NO_IDENTIFICATION &&
	        				faceclass.identities.size() > 0) {
		            	if (f.getAttributes()[expressionclassifier.INDEX_IDENTITY] >= 0) tag +=
		            			faceclass.identities.get(f.getAttributes()[expressionclassifier.INDEX_IDENTITY]).getName();
		            	else tag += getResources().getString(R.string.unknown);
		        	}
		        	
		        	switch (f.getAttributes()[expressionclassifier.INDEX_AGE]) {
		        	case expressionclassifier.AGE_YOUNG:
		        		paint.setColor(Color.CYAN);
		        		break;
		        	case expressionclassifier.AGE_MIDDLEAGED:
		        		paint.setColor(Color.MAGENTA);
		        		break;
		        	case expressionclassifier.AGE_OLD:
		        		paint.setColor(Color.BLUE);
		        		break;
		        		default:break;
		        	}
		            
		        	/*paint.setColor(Color.WHITE);
		        	canvas.drawRect((float)r.tl().x-2, (float)r.tl().y-2,
        					(float)r.br().x+2, (float)r.br().y+2, paint);*/
		        	
	            	if (f.getAttributes()[expressionclassifier.INDEX_GENDER] == expressionclassifier.GENDER_MALE) {
	            		canvas.drawRect((float)r.tl().x-2, (float)r.tl().y-2,
			        					(float)r.br().x+2, (float)r.br().y+2, paint);
	            	} else if (f.getAttributes()[expressionclassifier.INDEX_GENDER] == expressionclassifier.GENDER_FEMALE) {
	            		int size = (int)(r.br().x - r.tl().x)/2;
	            		canvas.drawCircle((float)r.tl().x + size, (float)r.tl().y + size, (float)size, paint);
	            	} else {
	            		canvas.drawLine((float)r.tl().x, (float)r.tl().y, (float)r.br().x, (float)r.br().y, paint);
	            	}
	            	
	            	for (Identity ID : faceclass.identities) {
	            		if (f.getAttributes()[expressionclassifier.INDEX_IDENTITY] == ID.getID()) {
	            			canvas.drawText(ID.getName(),(float)r.tl().x, (float)r.tl().y-20, paint);
	            			continue;
	            		}
	            	}	            	
	            	
	            	//canvas.drawText(expressionclassifier.expString(f.getAttributes()[expressionclassifier.INDEX_EXPRESSION]),(float)r.tl().x-10, (float)r.br().y+30, paint);
		        }
		        
		        // draw Neighborhood graph
		        paint.setColor(Color.WHITE);
		        for (Pair<Point_dt,Point_dt> edge: thisPhoto.RNG) {
					if (edge != null) if (edge.first != null && edge.second != null) {
						canvas.drawLine((float)edge.first.x(),(float)edge.first.y(),
							(float)edge.second.x(),(float)edge.second.y(),paint);
					}
				}
		        
		        //helper.savePicture(null, bmap, false, "canvas");
			}
		});
		
		Button allButton = (Button)view.findViewById(R.id.Button_ALL);
		allButton.setOnClickListener(new OnClickListener() {
			@Override public void onClick(View v) {
				addUsers();
			}
		});
		
		Button matchButton = (Button)view.findViewById(R.id.Button_MATCH);
		matchButton.setOnClickListener(new OnClickListener() {
			@Override public void onClick(View v) {
				showSimilarPhotos();
			}			
		});		
		
		Button calcMatchButton = (Button)view.findViewById(R.id.Button_CALC_MATCHES);
		calcMatchButton.setOnClickListener(new OnClickListener() {
			@Override public void onClick(View v) {
								
				//trainPhotoGroupClassifier();
				
				calculatePhotoSimilarity();
			}
		});
		
		Button saveButton = (Button)view.findViewById(R.id.Button_SAVE);
		saveButton.setOnClickListener(new OnClickListener() {
			@Override public void onClick(View v) {
				saveCurrent();
			}
		});
		
		Button setWeightsButton = (Button)view.findViewById(R.id.Button_SET_WEIGHTS);
		setWeightsButton.setOnClickListener(new OnClickListener() {
			@Override public void onClick(View v) {
				setWeights_fragment newFragment = new setWeights_fragment();
		        
				FragmentTransaction ft = getFragmentManager().beginTransaction();
		        Fragment prev = getFragmentManager().findFragmentByTag("setWeights_fragment");
		        if (prev != null) ft.remove(prev);
		        ft.setTransition(FragmentTransaction.TRANSIT_FRAGMENT_OPEN);
		        ft.addToBackStack(null);
		        
		        before.clear();
		        for (int g=0;g<similarity_weights.size();g++) before.add( similarity_weights.get(g) );
		        
		        newFragment.setArgs(similarity_weights);
		        newFragment.setListener(this_class);
		        newFragment.show(ft, "setWeights_fragment");
			}
		});
		
		Button exitButton = (Button)view.findViewById(R.id.Button_EXIT);
		exitButton.setOnClickListener(new OnClickListener() {
			@Override public void onClick(View v) {
				mCV.enableView();
				DialogFragment dialogFragment =
						(DialogFragment)getFragmentManager().findFragmentByTag("GalleryScanner");
				
				if (dialogFragment != null) dialogFragment.dismiss();
			}
		});
		
		nextButton.performClick();
        return view;
    }
    
    protected void saveCurrent() {
		// saves the photo currently drawn in canvas
    	Log.d(TAG, "debug: saving picture");
    	
    	helper.savePicture(null, bmap, false, "manual_save_");
	}

	@Override
	public void sendWeights(List<Integer> weights) {
    	boolean theSame = true;
    	for (int h=0;h<weights.size();h++) if (weights.get(h) != before.get(h)) theSame=false;
    	// .equals, pfft
    	
    	if (!theSame) {
    		similarity_weights = weights;
    		AlertDialog.Builder builder = new AlertDialog.Builder(getActivity());
        	
    		builder.setTitle("Retrain similarities?");
        	builder.setMessage("Current results no longer match the selected weights. Would you like to retrain?");
    		builder.setNegativeButton("No", new DialogInterface.OnClickListener() { 
    			@Override
    			public void onClick(DialogInterface dialog, int id) {
    				dialog.cancel();
    			}
    		});
    		builder.setPositiveButton("Yes", new DialogInterface.OnClickListener() {
    			@Override
    			public void onClick(DialogInterface dialog, int id) {
    				calculatePhotoSimilarity();
    				
    				dialog.cancel();
    			}
    		});
    		AlertDialog retrainQuestionDialog = builder.create();
    		retrainQuestionDialog.show();
    	}
    	
    	String s="";
    	for (int h=0;h<weights.size();h++) s+= Integer.toString(weights.get(h)) + " ";
    	Log.d(TAG, "debug: similarity weights set to "+s);
    }
    
    protected void showSimilarPhotos() {
    	
    	List<Similarity> in = similarPhotos.get(files[currentPhoto].getAbsolutePath());
    	if ( in != null ) {
	    	showMatch_fragment newFragment = new showMatch_fragment();
	        
			FragmentTransaction ft = getFragmentManager().beginTransaction();
	        Fragment prev = getFragmentManager().findFragmentByTag("showMatch_fragment");
	        if (prev != null) ft.remove(prev);
	        ft.addToBackStack(null);
	        
	        newFragment.setArgs(in);
	        newFragment.show(ft, "showMatch_fragment");
    	} else {
    		Toast.makeText(getActivity(), "No similar photos in file.", Toast.LENGTH_LONG).show();
    		Log.d(TAG, "fcp:"+files[currentPhoto]);
    	}
	}
    
	protected void calculatePhotoSimilarity() {
		similarPhotos.clear();
		
		updateImageData();
		
		Map<String,Photo> photos = new HashMap<String,Photo>();
		for (File file: files) {
			photos.put(file.getAbsolutePath(), new Photo(file.getAbsolutePath(), faceclass, true, true));
		}

    	for (File base: files) {
			thisPhoto = photos.get(base.getAbsolutePath()); //new photo(base.getAbsolutePath(), faceclass, true, loadGabors);
			List<Similarity> thisPhotoSimilarity = new ArrayList<Similarity>();
			
			if (thisPhoto.faces.size() < 2) {
				// Not a group so no point to calculate a similarity score against any other picture
				thisPhotoSimilarity.add( new Similarity( Double.MAX_VALUE, base.getAbsolutePath() ));
				similarPhotos.put(base.getAbsolutePath(), thisPhotoSimilarity );
				continue;
			}
			
			Log.d(TAG,"findSimilarPhotos, "+thisPhoto.faces.size()+" faces");
			
			// compare against every other photo
			for (File file: files) {
				double photoMatchValue = 0;
				if ( file.equals(base) ) continue; // skip self hit
				
				// check if this similarity is already calculated
				/*for (Entry<String, List<Similarity>> e : similarPhotos.entrySet()) {
					for (Similarity s : e.getValue()) {
						if ( ( file.getAbsolutePath().equals(e.getKey()) && base.getAbsolutePath().equals(s.string) ) ||
						   ( base.getAbsolutePath().equals(e.getKey()) && file.getAbsolutePath().equals(s.string) ) ) {
							photoMatchValue = s.score;
							thisPhotoSimilarity.add( new Similarity( photoMatchValue, file.getAbsolutePath() ));
						}
					}
				}*/
				
				Photo compare = photos.get(file.getAbsolutePath());
				
				if (!compare.found || compare.faces.size() < 2) {
					thisPhotoSimilarity.add( new Similarity( Double.MAX_VALUE, file.getAbsolutePath() ));
				} else if (photoMatchValue == 0) {
					// Calculates graph and group similarities based on "Efficient graph based spatial face context
			    	// representation and matching", which in turn is based on doctoral thesis "Graph matching and its
			    	// application in computer vision and bioinformatics" by Mikhail Zaslavskiy.
						
					// Calculate permutation matrix for property similarity matching
					//Log.d(TAG, "debug: sim_context_match for "+thisPhoto.path);
					//Log.d(TAG, "debug: thisPs = "+thisPhoto.faces.size()+" faces, cmp = "+compare.faces.size()+" faces");
					
					Photo larger = ( thisPhoto.faces.size() >= compare.faces.size() ) ? thisPhoto : compare;
					Range thisRange = new Range(0, larger.faces.size());
					
					Mat C = Mat.ones(larger.faces.size(), larger.faces.size(), CvType.CV_32FC1);
					Mat Cholder = Mat.zeros(C.size(), C.type());
					
					Mat similarity = new Mat(larger.faces.size(), larger.faces.size(), CvType.CV_32F);
											
					for (int k = 0;k<thisPhoto.faces.size();k++) {
						for (int j = 0;j<compare.faces.size(); j++) {
							
							double cost_value =	Imgproc.compareHist(thisPhoto.faces.get(k).gv, compare.faces.get(j).gv, Imgproc.CV_COMP_CORREL);
							if (cost_value > faceclass.GVLBPReject) {
								C.put(k, j, 1);
								continue;  
							} else cost_value = 0;
							
							// ID match is way more important than attribute match so small penalties

							// weights = 0-10
							if (thisPhoto.faces.get(k).attributes[expressionclassifier.INDEX_EXPRESSION] !=
									compare.faces.get(j).attributes[expressionclassifier.INDEX_EXPRESSION])
								cost_value += 0.33 *similarity_weights.get(expressionclassifier.INDEX_EXPRESSION);
								cost_value *= expressionclassifier.exp_dists[thisPhoto.faces.get(k).attributes[expressionclassifier.INDEX_EXPRESSION]]
								 	[compare.faces.get(j).attributes[expressionclassifier.INDEX_EXPRESSION]];
								 	
							if (thisPhoto.faces.get(k).attributes[expressionclassifier.INDEX_GENDER] !=
									compare.faces.get(j).attributes[expressionclassifier.INDEX_GENDER])
								cost_value += 0.2 * similarity_weights.get(expressionclassifier.INDEX_GENDER);
							if (thisPhoto.faces.get(k).attributes[expressionclassifier.INDEX_AGE] !=
									compare.faces.get(j).attributes[expressionclassifier.INDEX_AGE])
								cost_value += 0.2 * similarity_weights.get(expressionclassifier.INDEX_AGE);
							
							C.put(k, j, cost_value);
						}
					}
					Mat thisAdj = thisPhoto.adj.colRange(thisRange).rowRange(thisRange).clone();
					Mat compAdj = compare.adj.colRange(thisRange).rowRange(thisRange).clone();
																 
					// Spectral method
					/*MatOfFloat thisEigen = new MatOfFloat();
					MatOfFloat compEigen = new MatOfFloat();
					// Get eigenvectors
					Core.eigen(thisAdj, true, new Mat(), thisEigen);
					Core.eigen(compAdj, true, new Mat(), compEigen);
					
					// abs(eigens)
					Core.absdiff(thisEigen, Scalar.all(0), thisEigen);
					Core.absdiff(compEigen, Scalar.all(0), compEigen);
					
					// abs(Ug)abs(Uh^t)
					Core.gemm(thisEigen, compEigen, 1, new Mat(), 0, compEigen, Core.GEMM_2_T);
					
					double[][] weights = new double[thisEigen.height()][thisEigen.width()];
					for (int i = 0; i < thisEigen.height(); i++) {
						for (int j = 0; j < thisEigen.width(); j++) {
							weights[i][j] = 1 - thisEigen.get(i, j)[0];
							// hungarian matches minimums
						}
					}
					Mat P = new Mat(C.size(), C.type());
					Hungarian P_hun = new Hungarian( weights );
					int[] best_match = P_hun.execute();
					for (int i = 0; i < best_match.length; i++) P.put(i,best_match[i],1); */
											
					ArrayList<Integer> vals = new ArrayList<Integer>();
					ArrayList<Mat> Ps = new ArrayList<Mat>();
											
					for (int i=0;i<larger.faces.size();i++) vals.add(i);
					Permute<Integer> perm = new Permute<Integer>(vals);
																
					double bestMatch = Double.MAX_VALUE;
					List<Integer> permSeq = null;
					boolean okPerm = true; // the first permutation is the initial vector.. 1=1, 2=2 etc
										
					while(okPerm) {
						permSeq = perm.get_next();
						
						Mat P = Mat.zeros(larger.faces.size(), larger.faces.size(), CvType.CV_32F);
						// generate permutation matrix candidate from permSeq
						for (int j=0; j<permSeq.size();j++) {
							// value in permSeq(i) tells the index of test matching on Ps row
							if (permSeq.get(j) >= 0) P.put(j, permSeq.get(j), 1);
						}
													
						//Similarity value calculation
						// sim=PE^y
						// sim=(PE^y)P^t
						// sim=E^x - P E^y P^t
						Core.gemm(P,compAdj,1,new Mat(),0,similarity); 
						Core.gemm(similarity,P,1,new Mat(),0,similarity,Core.GEMM_2_T); 
						Core.subtract(thisAdj, similarity, similarity); 
						
						//Frobenius norm
						Core.gemm(similarity,similarity,1, new Mat(),0,similarity,Core.GEMM_2_T); // E E^t
						double structureSim = Core.sumElems(similarity.diag()).val[0]; // trace(E E^t)
																				
						//attribute similarity trace(C^t P)
						Core.gemm(C,P,1,new Mat(),0,Cholder,Core.GEMM_1_T);
						double label_match = Core.sumElems( Cholder.diag() ).val[0];
					
						double alpha = 0.5; // mixing parameter
						//Log.d(TAG, "debug: matchValue = "+photoMatchValue+", label match:"+label_match);
						photoMatchValue = (1-alpha)*structureSim + alpha*label_match;
						
						if (photoMatchValue <= bestMatch) {
							if (photoMatchValue < bestMatch) Ps.clear();
							
							bestMatch = photoMatchValue;
							Ps.add(P.clone());
							//Log.d(TAG, "matching "+file.getName()+" and "+base.getName());
							//Log.d(TAG, "debug Ps size:"+Ps.size()+" match SS:"+structureSim+", label="+label_match+" = "+photoMatchValue);
						}
						
						okPerm = perm.next_permutation();
					}
					
					// Ps should contain the best permutations by structure and label matching
					//int this_big_dim = (thisPhoto.picHeight > thisPhoto.picWidth) ?  thisPhoto.picHeight : thisPhoto.picWidth;
					//int compare_big_dim = (compare.picHeight > compare.picWidth) ?  compare.picHeight : compare.picWidth;
											
					// Pull vertex lists to vector form
					float this_x[] = new float[larger.faces.size()];
					float this_y[] = new float[larger.faces.size()];
					for (int i = 0; i < larger.faces.size(); i++) {
						if (i < thisPhoto.faces.size()) {
							this_x[i] = (float)thisPhoto.faces.get(i).location.x / thisPhoto.picWidth; //(float)this_big_dim;
							this_y[i] = (float)thisPhoto.faces.get(i).location.y / thisPhoto.picHeight; // (float)this_big_dim;
						}
					}						
					//Log.d(TAG, "debug this_x : "+ helper.FtoString(this_x));
					//Log.d(TAG, "debug this_y : "+ helper.FtoString(this_y));
					
					Mat comp_x_orig = Mat.zeros(larger.faces.size(), 1, CvType.CV_32F);
					Mat comp_y_orig = Mat.zeros(larger.faces.size(), 1, CvType.CV_32F);
					for (int i = 0; i < compare.faces.size(); i++) {
						comp_x_orig.put(i, 0, (float)compare.faces.get(i).location.x / compare.picWidth); //(float)compare_big_dim);
						comp_y_orig.put(i, 0, (float)compare.faces.get(i).location.y / compare.picHeight); //(float)compare_big_dim);
					}
					//Log.d(TAG, "debug compx "+comp_x.t().dump());
					//Log.d(TAG, "debug compy "+comp_y.t().dump());
					
					Mat comp_x = Mat.zeros(larger.faces.size(), 1, CvType.CV_32F);
					Mat comp_y = Mat.zeros(larger.faces.size(), 1, CvType.CV_32F);
					double bestShape = Double.MAX_VALUE;
					
					for (Mat P: Ps) {
						Core.gemm(P,compAdj,1,new Mat(),0,similarity); 
						Core.gemm(similarity,P,1,new Mat(),0,similarity,Core.GEMM_2_T);
																			
						// Transform best match from compare to "thisPhoto-space", x' = P^t y
						Core.gemm( P, comp_x_orig, 1, new Mat(), 0, comp_x, Core.GEMM_1_T );
						Core.gemm( P, comp_y_orig, 1, new Mat(), 0, comp_y, Core.GEMM_1_T );
						
						float[] t_comp_x = new MatOfFloat( comp_x ).toArray();
						float[] t_comp_y = new MatOfFloat( comp_y ).toArray();
						
						//Log.d(TAG, "debug best x: "+helper.FtoString(this_x)+" vs "+helper.FtoString(t_comp_x));
						//Log.d(TAG, "debug best y: "+helper.FtoString(this_y)+" vs "+helper.FtoString(t_comp_y));
						
						double shape = 0;
						for (int y = 0; y < similarity.height()-1; y++) {							
							for (int x = y; x < similarity.width(); x++) {
								// if the graph being compared againt has an edge here
								// calculate orientation difference
								//if ((thisAdj.get(y,x)[0] > 0) && 
								//if (similarity.get(y,x)[0] > 0) {
						// diff 1 = vertex1 in thisPhoto - vertex2 in this Photo
						// diff 2 = v1 from best match from compare - v2 from best match from compare
						// similarity of O = norm( (  diff1 / norm ( diff1) ) - ( diff2 / norm( diff2 ) ) )

									Point_dt anchor1 = new Point_dt( this_x[y],  this_y[y] );
									
									Point_dt diff1 = new Point_dt ( (this_x[y] - this_x[x]),
																  	(this_y[y] - this_y[x]) );
									
									Point_dt anchor2 = new Point_dt( t_comp_x[y],  t_comp_y[y] );
																			
									Point_dt diff2 = new Point_dt ( (t_comp_x[y] - t_comp_x[x]),
																    (t_comp_y[y] - t_comp_y[x]) );
									
									double diff1norm = diff1.distance(anchor1);
									double diff2norm = diff2.distance(anchor2);
									
									if (diff1norm > 0) {
										diff1.setX( diff1.x() / diff1norm );
										diff1.setY( diff1.y() / diff1norm );
									} else {
										diff1.setX( 0 );
										diff1.setY( 0 );
									}
									
									if (diff2norm > 0) {
										diff2.setX( diff2.x() / diff2norm );
										diff2.setY( diff2.y() / diff2norm );
									} else {
										diff2.setX( 0 );
										diff2.setY( 0 );
									}
									
									//Log.d(TAG, "debug diff1=("+diff1.y()+","+diff1.x()+"), diff2="+diff2.y()+","+diff2.x());
									
									// diff1 is recycled here for the whole inside part
									diff1.setX( diff1.x() - diff2.x() );
									diff1.setY( diff1.y() - diff2.y() );
									
									//Log.d(TAG, "debug norm = "+diff1.distance(diff2));
									
									if (!Double.isNaN(diff1.distance(diff2)))
										shape += diff1.distance(diff2); // norm
								//}
							}
						}
						if (shape < bestShape) bestShape = shape;
						
					}
					//Log.d(TAG, "debug: "+thisPhoto.path+" vs "+compare.path+", graph = "+photoMatchValue+", shape = "+bestShape);
					
					thisPhotoSimilarity.add( new Similarity( bestShape, file.getAbsolutePath() ));
				}
			}
			
			Collections.sort( thisPhotoSimilarity );
			similarPhotos.put(base.getAbsolutePath(), thisPhotoSimilarity);
						
			Log.d(TAG, "debug: File "+base.getAbsolutePath()+ " similarities");
			for (Similarity s: thisPhotoSimilarity) Log.d(TAG, s.score + " --- " + s.string);
			
		}
    	saveSimilarities();
	}
	
	protected void trainPhotoGroupClassifier() {
		selectedIdx = new ArrayList<Integer>();
		ArrayList<Integer> selectedTestIdx = new ArrayList<Integer>();
		ArrayList<Integer> dividers = new ArrayList<Integer>();
		
		double best = 0;
		
		// open results file for writing
		try {
			File file = new File(identRootDir.getAbsolutePath()+"/kNNtest.txt");
			if (!file.exists()) file.createNewFile();
			FileWriter fw = new FileWriter(file, false);
			BufferedWriter bw = new BufferedWriter(fw);
		
		// Use pairwise distances as feature patterns for the kNN ?
    	// This part was rather poorly explained in the paper.    	
    	
		ArrayList<Pair<String, ArrayList<double[]>>> allGroups = new ArrayList<Pair<String, ArrayList<double[]>>>();
    	
		dividers.add( 0 );
    	// Read the "The Images of Groups Dataset" data as truth.
    	loadGroupData("Fam2", allGroups, GROUP_FAMILY);
    	loadGroupData("Fam4", allGroups, GROUP_FAMILY);
    	loadGroupData("Fam5", allGroups, GROUP_FAMILY);
    	loadGroupData("Fam8", allGroups, GROUP_FAMILY);
    	dividers.add( allGroups.size() );
    	loadGroupData("Group2", allGroups, GROUP_GROUP);
    	loadGroupData("Group4", allGroups, GROUP_GROUP);
    	loadGroupData("Group5", allGroups, GROUP_GROUP);
    	loadGroupData("Group8", allGroups, GROUP_GROUP);
    	dividers.add( allGroups.size() );
    	loadGroupData("Wed2", allGroups, GROUP_WEDDING);
    	loadGroupData("Wed3", allGroups, GROUP_WEDDING);
    	loadGroupData("Wed5", allGroups, GROUP_WEDDING);
    	dividers.add( allGroups.size() );
    	
		for (int numberOfTrainingSamplesPerClass = 10; numberOfTrainingSamplesPerClass < 150; numberOfTrainingSamplesPerClass+=5 ) {
			selectedIdx.clear();
			selectedTestIdx.clear();
			
	    	// pick training and test data equally, 100 test samples per class
	    	for (int n=0;n<dividers.size()-1;n++) {
	    		selectRandomIndecies(dividers.get(n), dividers.get(n+1), selectedIdx, numberOfTrainingSamplesPerClass, selectedTestIdx);
	    		selectRandomIndecies(dividers.get(n), dividers.get(n+1), selectedTestIdx, 100, selectedIdx);
	    	}
	    	
	    	/*Collections.sort(selectedIdx);
	    	Log.d(TAG, "debug selected idx:"+selectedIdx+" = "+selectedIdx.size()+" elements");
	    	Log.d(TAG, "debug selected test idx:"+selectedTestIdx);*/
	    	
	    	ArrayList<Pair<String, ArrayList<double[]>>> test_groups = new ArrayList<Pair<String, ArrayList<double[]>>>();
	    	for (int i=0;i<selectedIdx.size();i++) {
	    		test_groups.add(allGroups.get(selectedIdx.get(i))); // pick selected to be used to compare
	    	}
	    	
	    	Mat kNNdata = Mat.zeros(test_groups.size(), test_groups.size(), CvType.CV_32F);
	    	Mat kNNresponses = Mat.zeros(test_groups.size(),1, kNNdata.type());
	    	
	    	// run through every pairing and build a distance matrix ready to be chopped down
	    	// row by row for kNN training, process upper triangle and mirror it
	    		    	
	    	for (int i=0;i<test_groups.size()-1;i++) {
	    		kNNresponses.put(i, 0, test_groups.get(i).second.get(0)[4]); //stupid hitch hiking
	    		for (int j=i+1;j<test_groups.size();j++) { // diag distance = naturally 0
	    			float photoMatchValue = matchGroups(test_groups.get(i).second, test_groups.get(j).second);
	    			
	    			kNNdata.put(i, j, photoMatchValue); //distances to and fro
	    			kNNdata.put(j, i, photoMatchValue);
	    		}
			}
	    	
	    	CvKNearest test_similarity_knn = new CvKNearest();
	    	test_similarity_knn.train(kNNdata, kNNresponses);
	    	   	
	    	//Build test sample list-matrix and correct answers
	    	Mat testMat = new Mat(selectedTestIdx.size(), selectedIdx.size(), CvType.CV_32F);
	    	Mat corr = new Mat();
	    	for (int i=0;i<selectedTestIdx.size();i++) {
	    		corr.push_back(new MatOfFloat( (float)allGroups.get(selectedTestIdx.get(i)).second.get(0)[4] ));
	    		
	    		for (int j=0;j<selectedIdx.size();j++) {
	    			double photoMatchValue = matchGroups(test_groups.get(j).second, allGroups.get(selectedTestIdx.get(i)).second);
	    			testMat.put(i,j,photoMatchValue);
	    		}
	    	}
	    	
	    	// aaand match
	    	Mat diff = Mat.zeros(kNNresponses.size(), kNNresponses.type());
	    	Mat results = Mat.zeros(diff.size(), diff.type());
	    	for (int k=2;k<15;k++) {
	    		test_similarity_knn.find_nearest(testMat, k, results, new Mat(), new Mat());
	    		    		
	    		Core.absdiff(corr, results, diff);
	    		Imgproc.threshold(diff, diff, 0,1, Imgproc.THRESH_BINARY_INV);
	    		// if |correct-recognized| > 0 set it to 0, 1 if diff=0, which is correct classification
	    		float corrpr = (float)(Core.sumElems(diff).val[0] / diff.size().height) * 100;
	    		if (corrpr > best) {
	    			similarity_knn = test_similarity_knn;
	    			groups = test_groups;
	    			best = corrpr; // save the best classifier afterwards
	    		}    		
	    		
	    		Log.d(TAG, "debug: kNN("+k+") correctness="+corrpr);
	    		bw.write("NoTSpC:"+Integer.toString(numberOfTrainingSamplesPerClass)+", k="+Integer.toString(k)+" = "+Float.toString(corrpr));
	    		bw.newLine();
	    	}
	    	bw.write("\n----\n\n");
		}
		bw.write("Best result: "+Double.toString(best));
    	
    	bw.close();
		fw.close();
	} catch (Exception e) {
		e.printStackTrace();
		helper.crash();
	}
	
	}
	
	private int selectRandomIndecies(int bound, int size,
			List<Integer> selected, int randoms, List<Integer> avoid) {
		
		Random rand = new Random(System.currentTimeMillis());
		int randomIdx;
		
		for (int i=0;i<randoms;i++) {
			do {
				randomIdx = rand.nextInt(size - bound) + bound;
			} while ( avoid.contains(randomIdx) );
			
			selected.add(randomIdx);
		}
		return size;
	}

	private float matchGroups( ArrayList<double[]> first, ArrayList<double[]> second) {
		int face_count_difference = Math.abs(first.size() - second.size());
		
		double[][] links = calculateBipartitionWeights(first, second);
		
		//for (int k =0;k<links.length;k++) Log.d(TAG, helper.DtoString(links[k]));
						
		// Now we should have the full link cost table between current and under comparison photos.
		// To select the best links the Hungarian algorithm is applied
		
		Hungarian hungarian = new Hungarian( links );
		int[] selected = hungarian.execute();
		
		// This version will return list fitting one row to one column, so selected[0] is the best
		// match for face0 in the photo we are looking matches for. O(n^3) speed
		//Log.d(TAG, "debug: selected = "+helper.ItoString(selected));
		float photoMatchValue = 0;
		
		for (int k=0; k<selected.length; k++) {
			photoMatchValue += (float)links[k][selected[k]];
		}
		// Finally, add the penalty for dissimilar number of faces
		photoMatchValue += similarity_weights.get(4)*face_count_difference;
		return photoMatchValue;
	}

	private void loadGroupData(String filename, ArrayList<Pair<String,ArrayList<double[]>>> group, int type) {
		Log.d(TAG, "debug: loading Group data "+filename);
		try {
			FileReader reader = new FileReader(new File(identRootDir.getAbsolutePath()+"/Groupdata/"+filename+"sizes.txt"));
			Scanner scan1 = new Scanner( reader );
			Pattern intPattern = Pattern.compile("\\d+");
			
			Map<String, Point> sizes = new HashMap<String,Point>();
			while (scan1.hasNext()) {
				String name = scan1.next();
				String rest = scan1.nextLine();
				Matcher match = intPattern.matcher(rest);
				match.find(); int w = Integer.parseInt( match.group(0) );
				match.find(); int h = Integer.parseInt( match.group(0) );
				
				Point p = new Point( w,h );
				
				sizes.put(name, p);
			}
			scan1.close();
			reader.close();
			
			FileReader fileReader = new FileReader(new File(identRootDir.getAbsolutePath()+"/Groupdata/"+filename+"a.txt"));
        	Scanner scan = new Scanner( fileReader );
        	String thisName="";
        	
        	ArrayList<double[]> thisFaces = new ArrayList<double[]>();
        	double totX=0, totY=0;
        	Point thisSize = new Point();
        	        			
        	while (scan.hasNext()) {
	        	String input = scan.nextLine();
	        	
	        	if ( input.endsWith(".jpg") ) { // this is an "IMAGE" row
	        		if (thisFaces.size() > 0) {
		        		for (double[] d: thisFaces) {
		        			d[0] -= (totX / thisFaces.size());
		        			d[1] -= (totY / thisFaces.size()); //mean remove
		        		}
		        		group.add(new Pair<String,ArrayList<double[]>>(thisName, thisFaces));
	        		}
	        		
	        		thisName = input;
	        		thisSize = sizes.get(thisName);
	        		thisFaces = new ArrayList<double[]>(); totX=0; totY=0;
	        	} else { // an "FACE" row
	        		Matcher match = intPattern.matcher(input);

	        		match.find(); double leftX = (double)Integer.parseInt(match.group(0)) / thisSize.x;
	        		match.find(); double leftY = (double)Integer.parseInt(match.group(0)) / thisSize.y;
	        		match.find(); double rightX = (double)Integer.parseInt(match.group(0)) / thisSize.x;
	        		match.find(); double rightY = (double)Integer.parseInt(match.group(0)) / thisSize.y;
	        		match.find(); int age = Integer.parseInt(match.group(0));
	        		match.find(); int gender = Integer.parseInt(match.group(0));
	        		 
	        		// guestimation conversion to face box center.
	        		// Head width ~ 5*eye width
	        		// Head height ~ 7*eye width, eyeY ~ between 3rd and 4th width
	        		float eyeW = ((float)(rightX - leftX) / 2.0f);
	        		
	        		leftX = leftX + eyeW; // ~centerx
	        		leftY = leftY - (eyeW / 2); // ~centerY
	        			        		
	        		double[] face = new double[]{ leftX, leftY, age, gender, type };
	        		//pointless repetition for type, but makes it easier to implement via map
	        		totX += leftX; totY += leftY;
	        		
	        		thisFaces.add(face);
	        	}
        	}
        	scan.close();
        	fileReader.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	// scale and normalize the locations in this photo to match the given scale
    /*protected void scaleToMean(double scaleTo) {
    	
    	if (faces.size() == 0) return;
    	
    	double comp_mean_face_size = 1;
    	if (scaleTo != 1) {
	    	for (Face f: faces) {
	    		comp_mean_face_size += f.location.width;
	    	}
	    	comp_mean_face_size /= faces.size();
    	}
    	
    	double scaling_factor = scaleTo / comp_mean_face_size;
    	//Log.d(TAG, "scaling_factor = "+scaling_factor);
    	
    	double meanX=0,meanY=0;
    	for (int i=0;i<faces.size();i++) {
    		meanX += faces.get(i).location.x;
    		meanY += faces.get(i).location.y;
    	}
    	meanX /= faces.size();
    	meanY /= faces.size();
    	
    	// rounding to integers won't matter that much
    	for (Face f: faces) {
    		// scale the faces with scaling factor (same as scaling the whole picture)
    		// and remove the mean
    		f.location.x = (int)( f.location.x * scaling_factor - meanX);
    		f.location.y = (int)( f.location.y * scaling_factor - meanY);
    		
    		f.location.width = (int)( f.location.width * scaling_factor );
    		f.location.height = (int)( f.location.height * scaling_factor );
    		
    		Log.d(TAG, "new face, scale="+scaling_factor+" : (y,x,h,w):"+f.location.y +" "+ f.location.x + " " + f.location.height+" "+ f.location.width );
    	}
    }*/
	

	// weights = distance_alpha, exppression_coeff, gender_coeff, age_coeff, face_count_gamma
	private double[][] calculateBipartitionWeights(ArrayList<double[]> first,
												   ArrayList<double[]> second) {
		if (similarity_weights.size() != 5) {
			Log.e(TAG, "weights.size != 5");
			return new double[0][0];
		}
		
		int smaller = (first.size() <= second.size()) ? first.size() : second.size();
		int larger = (first.size() > second.size()) ? first.size() : second.size();
		double[][] retval = new double[smaller][larger];
		
		// Every face in photo with fewer faces is matched against faces in the other photo.
		// Depending on weights every link gets a match cost value. Lower value means a better match
		for (int i = 0; i < smaller; i++) {
			for (int j = 0; j < larger; j++) {
				if (first.size() <= second.size() ) {
					retval[i][j] = calculateLinkCost( first, second, i, j );
				} else {
					retval[i][j] = calculateLinkCost( first, second, j, i ); 
				}
			}
		}
						
		return retval;
	}

	// i = index in base photo, j = index in comp
	private double calculateLinkCost(ArrayList<double[]> first,
									 ArrayList<double[]> second, int i, int j) {
		double dx = first.get(i)[0] - second.get(j)[0];
		double dy = first.get(i)[1] - second.get(j)[1];
		
		double distance = Math.sqrt(Math.pow(dx, 2) + Math.pow(dy, 2)) * similarity_weights.get(0);
				
		double age = Math.abs(first.get(i)[2] - second.get(j)[2])
				* similarity_weights.get(expressionclassifier.INDEX_AGE);
		
		double gender = Math.abs(first.get(i)[3] - second.get(j)[3])
				* similarity_weights.get(expressionclassifier.INDEX_GENDER);
		
		return distance + gender + age;
	}
	
	// save the best matches for each photo
	private void saveSimilarities() {
		try {
			File file = new File(matchesFile);
			if (!file.exists()) file.createNewFile();
			
			FileWriter fw = new FileWriter(file, false);
			BufferedWriter bw = new BufferedWriter(fw);
			
			bw.write(Integer.toString(similarity_weights.size())+delim);
			for (int i: similarity_weights) {
				bw.write(Integer.toString(i) + delim);
			}
			bw.newLine();

			for (Entry<String,List<Similarity>> sim: similarPhotos.entrySet()) {
				bw.write(sim.getKey() + delim);
				bw.write(Integer.toString(sim.getValue().size()) + delim);
				
				for (Similarity s: sim.getValue()) {
					// skip self matches
					bw.write(s.score + delim + s.string + delim);
				}
				bw.newLine();
			}
			bw.close();
			fw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
		
	private void loadSimilarities() {
		similarPhotos.clear();
			
		try {
			FileReader fileReader = new FileReader(new File(matchesFile));
        	Scanner scan = new Scanner( fileReader );
        	
        	int i = scan.nextInt();
        	for (int k=0;k<i;k++)
        		similarity_weights.add(scan.nextInt());
        	
        	while (scan.hasNext()) {
	        	String inKey = scan.next();
        		int sims = scan.nextInt();
        		List<Similarity> inList = new ArrayList<Similarity>();
        		
        		//Log.d(TAG, "debug inkey sims "+inKey+", "+sims);
	        		
	        	for (int c=0; c < sims; c++) {
	    			Similarity inPair = new Similarity( scan.nextDouble(), scan.next() );
	    			inList.add(inPair);
	    			//Log.d(TAG, "debug read: "+inPair.string+" = "+inPair.score);
	        	}
        		
	        	similarPhotos.put( inKey, inList );
        	}
        	
        	Log.d(TAG, "debug: Loaded "+similarPhotos.size()+" similarity lists");
        	
        	scan.close();
        	fileReader.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	// Iterate through the entire folder and save the results in photoInfoFile-file
	private void updateImageData() {
		try {
			File photoFile = new File(photoInfoFile);
			if (!photoFile.exists()) photoFile.createNewFile();
						
			FileWriter fw = new FileWriter(photoFile,false);
			BufferedWriter bw = new BufferedWriter(fw);
			
			for (File file : files) {
				Log.d(TAG, "file:"+file.getAbsolutePath());
				// Load reasonably down sampled version which will fit into RAM,
				// find faces and calculate graphs and adj. matrix
				Photo process = new Photo( file.getAbsolutePath(), faceclass, false, false );
				
				bw.write( file.getAbsolutePath() + delim);
				// The absolute path to this photo as ID in file
				
				bw.write( Integer.toString(process.picHeight) + delim + Integer.toString(process.picWidth) + delim );
				// image dimensions				
				
				bw.write( Integer.toString(process.faces.size()) + delim );
				Log.d(TAG, "debug: saving "+process.faces.size()+" faces");
				
				Log.d(TAG, "debug: faces = "+process.faces.size());
				for (int face = 0; face < process.faces.size(); face++) {
					Rect r = process.faces.get(face).location;
					Mat ROI = process.gray.submat(r).clone();

					// Write the location of this face
					bw.write( Integer.toString( r.x ) + delim);
					bw.write( Integer.toString( r.y ) + delim);
					bw.write( Integer.toString( r.width ) + delim);
					bw.write( Integer.toString( r.height ) + delim);
					
		        	int[] result;
		        	Mat hist = new Mat();
		        	
		        	result = faceclass.expclass.identifyExpression(ROI); // resizes to 64x64
		        	// gender, expression and age in order from expressionclassifier
		        	for (int j=1;j<4;j++) bw.write(Integer.toString(result[j]) + delim );
		        	
		        	Imgproc.resize(ROI, ROI, new Size(64,64));
		        	ROI = ROI.colRange(8, 56);
		        	
		        	faceclass.expclass.nativelib.GaborLBP_Histograms(ROI, hist, faceclass.LUT, 8, -1, -1);
		        	float[] gaborVals = new MatOfFloat(hist).toArray();

		        	bw.write( Integer.toString( gaborVals.length ) + delim);
		        	for (int h=0;h<gaborVals.length;h++) bw.write( Float.toString(gaborVals[h]) + delim );
		        			        	
		        	bw.newLine();
		        }
				
	        	// neighbor graph size
	        	bw.write( Integer.toString( process.RNG.size() ) + delim );
		        for (Pair<Point_dt,Point_dt> edge: process.RNG) {
		        	bw.write( Double.toString( edge.first.x() ) + delim );
		        	bw.write( Double.toString( edge.first.y() ) + delim );
		        	bw.write( Double.toString( edge.second.x() ) + delim );
		        	bw.write( Double.toString( edge.second.y() ) + delim );
				}
		        
		        // Save the adjacency matrix based on this face order
		        
		        for (int i=0;i<process.adj.size().height;i++)
		        for (int j=0;j<process.adj.size().width;j++) {
		        	double[] in = new double[3];
		        	in = process.adj.get(i, j);
		        	
		        	if (in[0] != 0) bw.write( Integer.toString(i)+delim+Integer.toString(j)+delim);
		        }
		        bw.write("-1"+delim+"-1"+delim+"-1"+delim+"-1");
				bw.newLine();
				
				process.gray = null;
			}
			
			bw.flush(); fw.flush();
			bw.close();	fw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	protected void addUsers() {
		int i = 0;
		thisPhoto.getFullSizedSamples();
		
		for (Face f : thisPhoto.faces)
	    {
	        addUser_fragment newFragment = new addUser_fragment();
	        
			FragmentTransaction ft = getFragmentManager().beginTransaction();
	        Fragment prev = getFragmentManager().findFragmentByTag("addUser_fragment");
	        if (prev != null) ft.remove(prev);
	        ft.addToBackStack(null);
	        	        
	        //Log.d(TAG, "debug "+ROI+" "+i+" - "+faceclass);
	        newFragment.setArgs(f.getFullPic(), i++, thisPhoto.faces.size(), faceclass);
	        newFragment.show(ft, "addUser_fragment");
	    }
	}

		
}
