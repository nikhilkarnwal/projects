package com.nikarn.securedata;

import android.content.Context;
import android.provider.ContactsContract;
import android.support.v4.content.ContextCompat;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import com.nikarn.securedata.Model.DataItem;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by nikarn on 30/03/2017.
 */

public class DataListAdapter extends RecyclerView.Adapter<DataListAdapter.ItemViewHolder> {
    private Context mContext;
    private List<DataItem> mItemList;

    public void addData(DataItem dataItem) {
        mItemList.add(dataItem);
        notifyItemInserted(mItemList.size()-1);
    }

    class ItemViewHolder extends RecyclerView.ViewHolder {
        TextView mHeading;
        TextView mContent;

        ItemViewHolder(View view) {
            super(view);
            mHeading = (TextView) view.findViewById(R.id.list_item_heading);
            mContent = (TextView) view.findViewById(R.id.list_item_content);
        }
    }

    public DataListAdapter(Context context) {
        this.mContext = context;
        mItemList = items;
    }

    @Override
    public ItemViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        View itemView = LayoutInflater.from(parent.getContext()).inflate(R.layout.list_item_data, parent, false);
        return new ItemViewHolder(itemView);
    }

    @Override
    public void onBindViewHolder(final ItemViewHolder holder, int position) {
        DataItem item = mItemList.get(position);
        holder.mHeading.setText(item.Name);
        //holder.mTextView.setTextColor(ContextCompat.getColor(mContext, R.color.white));
        holder.mContent.setText(item.Data);
        holder.itemView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                int position = holder.getAdapterPosition();
                actionOnDataItemClick(mContext, mItemList.get(position));
            }
        });
    }

    private void actionOnDataItemClick(Context mContext, DataItem dataItem) {
        
    }


    @Override
    public int getItemCount() {
        return mItemList.size();
    }
}