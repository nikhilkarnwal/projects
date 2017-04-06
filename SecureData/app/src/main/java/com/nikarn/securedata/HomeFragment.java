package com.nikarn.securedata;

import android.content.Context;
import android.net.Uri;
import android.os.Bundle;
import android.support.v4.app.Fragment;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;

import com.nikarn.securedata.Utils.Utility;


public class HomeFragment extends Fragment {
    public HomeFragment() {
        // Required empty public constructor
    }


    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {

        // Inflate the layout for this fragment
        final View rootView = inflater.inflate(R.layout.fragment_home, container, false);
        Button loginButton = (Button) rootView.findViewById(R.id.bttn_login);
        loginButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String passText = ((EditText) rootView.findViewById(R.id.edit_text_password)).getText().toString();
                validatePassword(passText);
            }
        });
    }

    private void validatePassword(String passText) {
        if (Utility.isValidPassword(passText)){

        }
    }
}
