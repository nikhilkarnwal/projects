package com.nikarn.securedata.Utils;

import android.util.Log;

import java.io.IOException;

/**
 * Created by nikarn on 3/31/2017.
 */
public class LogUtils {

    public static void logHandledException(IOException e) {
        Log.v("Exception", e.toString());
    }
}
