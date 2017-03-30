package com.nikarn.securedata.Utils;

import android.os.AsyncTask;

import com.nikarn.securedata.Listener.DataListListener;

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
    public static void readData(DataListListener dataListListener){
        AsyncTask<Void, Void, Void> task = new AsyncTask<Void, Void, Void>() {
            @Override
            protected Void doInBackground(Void... voids) {
                ArrayList<String>
                return null;
            }
        };
    }

    private static void readAppData(){

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
