package com.siperia.peopleinphotos;

import android.app.ProgressDialog;
import android.content.Context;
import android.os.AsyncTask;

public class ProgressbarDisplay extends AsyncTask {

	private Context context;
    private ProgressDialog Dialog = new ProgressDialog( context );
    
    public ProgressbarDisplay( Context context ) {
    	this.context = context;
    }
    
    protected void onPreExecute() {
        Dialog.setMessage("Updating face and group graph data...");
        Dialog.show();
    }
    
    protected void onPostExecute(Void unused) {
        Dialog.dismiss();
    }

	@Override
	protected Void doInBackground(Object... args) {
		// TODO Auto-generated method stub
		return null;
	}

}
