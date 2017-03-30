package com.nikarn.securedata.Utils;


import android.support.v4.app.Fragment;

import com.nikarn.securedata.DataListFragment;

/**
 * Created by nikarn on 3/31/2017.
 */

public class FragmentFactory {
    public static Fragment getFragment(FragmentPage fragmentPage){
        Fragment fragment = null;
        switch (fragmentPage){
            case DataListFragment:
                fragment = new DataListFragment();
                break;
            case DataContentFragment:
            default:
                fragment = new Fragment();
                break;
        }
        return fragment;
    }
}
