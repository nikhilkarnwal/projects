package com.nikarn.securedata;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import com.nikarn.securedata.libs.DataCipher;
import com.nikarn.securedata.libs.UserData;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button submit = (Button)findViewById(R.id.butn_submit);
        submit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                EditText editText = (EditText) findViewById(R.id.edit_key);
                UserData.getInstance().setPassword(editText.getText().toString());
                editText = (EditText) findViewById(R.id.edit_data);
                DataCipher dataCipher = new DataCipher();
                String dataToEncrypt = editText.getText().toString();
                Log.v("Data to encrypt",dataToEncrypt );
                byte[] data = dataCipher.encryptData(dataToEncrypt);
                TextView textView = (TextView) findViewById(R.id.text_data);
                textView.setText(dataCipher.decryptData(data));
            }
        });
        getSupportFragmentManager().beginTransaction().add
    }
}
