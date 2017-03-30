package com.nikarn.securedata;

import android.content.Context;
import android.net.Uri;
import android.os.Bundle;
import android.support.v4.app.Fragment;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import com.nikarn.securedata.Listener.DataListListener;
import com.nikarn.securedata.Model.DataItem;
import com.nikarn.securedata.Utils.Utility;

public class DataListFragment extends Fragment {

    public DataListFragment() {
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
        View root = inflater.inflate(R.layout.fragment_data_list, container, false);

        RecyclerView recyclerView = (RecyclerView) root.findViewById(R.id.list_view_data);
        recyclerView.setLayoutManager(new LinearLayoutManager(getContext(), LinearLayoutManager.VERTICAL, false));
        DataListAdapter dataListAdapter = new DataListAdapter(getContext());
        recyclerView.setAdapter(dataListAdapter);
        init(dataListAdapter);
        return root;
    }

    private void init(final DataListAdapter dataListAdapter) {
        Utility.readData(new DataListListener() {
            @Override
            public void addData(DataItem dataItem) {
                dataListAdapter.addData(dataItem);
            }
        });
    }


}
