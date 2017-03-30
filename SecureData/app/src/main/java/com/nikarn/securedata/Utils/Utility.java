package com.nikarn.securedata.Utils;

import android.os.AsyncTask;
import android.os.Environment;
import android.provider.ContactsContract;
import android.util.Log;

import com.google.gson.Gson;
import com.nikarn.securedata.Listener.DataListListener;
import com.nikarn.securedata.Model.AppData;
import com.nikarn.securedata.Model.DataItem;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by nikarn on 30/03/2017.
 */

public class Utility {
    public static void readData(final DataListListener dataListListener){
        AsyncTask<Void, DataItem, Void> task = new AsyncTask<Void, DataItem, Void>() {
            @Override
            protected Void doInBackground(Void... voids) {
                AppData appData = readAppData();
                for (String file :
                        appData.mFileNames) {
                    publishProgress(readDataFile(file));
                }
                return null;
            }

            @Override
            protected void onProgressUpdate(DataItem... values) {
                dataListListener.addData(values[0]);
            }
        };
    }

    private static DataItem readDataFile(String file) {
        Gson gson = new Gson();
        DataItem dataItem = gson.fromJson(readAllTextFile(new File(file)), DataItem.class);
        return dataItem;
    }

    private static AppData readAppData(){
        Gson gson = new Gson();
        AppData appData = gson.fromJson(readAllTextFile(getAppDataDirectory()),AppData.class);
        return appData;
    }

    private static File getAppDataDirectory() {
        return new File(Environment.getExternalStorageDirectory(), Constants.APPDATAFILE);
    }

    public static String readAllTextFile(File file) {
        StringBuilder stringBuilder = new StringBuilder();
        if (file.exists() && file.canRead()) {
            try {
                BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
                String line;
                while ((line = bufferedReader.readLine()) != null) {
                    stringBuilder.append(line);
                }
                bufferedReader.close();
            } catch (IOException e) {
                LogUtils.logHandledException(e);
            }
        }
        return stringBuilder.toString();
    }

    public static void writeAllTextFile(File file, String content) {
        File dirs = file.getParentFile();
        if (!dirs.exists()) {
            dirs.mkdirs();
        }
        try {
            if ((file.exists() || file.createNewFile())
                    && file.canWrite()) {
                BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(file));
                bufferedWriter.write(content);
                bufferedWriter.close();
            }
        } catch (IOException e) {
            LogUtils.logHandledException(e);
        }
    }
}
