package com.nikarn.securedata.libs;

import android.hardware.usb.UsbRequest;
import android.os.Environment;
import android.util.Log;

import java.io.BufferedOutputStream;
import java.io.Console;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.lang.reflect.Array;
import java.security.NoSuchAlgorithmException;
import java.security.spec.InvalidKeySpecException;
import java.security.spec.KeySpec;
import java.util.Arrays;
import java.util.logging.Logger;

import javax.crypto.Cipher;
import javax.crypto.SecretKey;
import javax.crypto.SecretKeyFactory;
import javax.crypto.spec.PBEKeySpec;
import javax.crypto.spec.SecretKeySpec;

/**
 * Created by nikarn on 3/29/2017.
 */

public class DataCipher {

    private static String algorithm = "AES";
    private byte[] encodeFile(SecretKey yourKey, byte[] fileData)
            throws Exception {
        byte[] data = yourKey.getEncoded();
        SecretKeySpec skeySpec = new SecretKeySpec(data, 0, data.length,
                algorithm);
        Cipher cipher = Cipher.getInstance(algorithm);
        cipher.init(Cipher.ENCRYPT_MODE, skeySpec);

        byte[] encrypted = cipher.doFinal(fileData);

        return encrypted;
    }

    private byte[] decodeFile(SecretKey yourKey, byte[] fileData)
            throws Exception {
        Cipher cipher = Cipher.getInstance(algorithm);
        cipher.init(Cipher.DECRYPT_MODE, yourKey);

        byte[] decrypted = cipher.doFinal(fileData);

        return decrypted;
    }

    public void encryptFile(String dataToSave, String fileToEncrypt) {
        try {
            File file = new File(Environment.getExternalStorageDirectory()
                    + File.separator, fileToEncrypt);
            BufferedOutputStream bos = new BufferedOutputStream(
                    new FileOutputStream(file));
            String password = UserData.getInstance().getPassword();
            SecretKey yourKey = generateKey(password.concat(password));
            byte[] filesBytes = encodeFile(yourKey, dataToSave.getBytes());
            Log.v("Data", decodeFile(yourKey, filesBytes).toString());
            bos.write(filesBytes);
            bos.flush();
            bos.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public byte[] encryptData(String dataToSave){
        String password = UserData.getInstance().getPassword();
        SecretKey yourKey = null;
        byte[] filesBytes = null;
        try {
            yourKey = generateKey(password.concat(password));
            filesBytes = encodeFile(yourKey, dataToSave.getBytes());
        } catch (NoSuchAlgorithmException e) {
            e.printStackTrace();
        } catch (InvalidKeySpecException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return filesBytes;
    }

    public String decryptData(byte[] data){
        String password = UserData.getInstance().getPassword();
        SecretKey yourKey = null;
        String ans = null;
        try {
            yourKey = generateKey(password.concat(password));
            ans = new String(decodeFile(yourKey, data), "UTF-8");
        } catch (NoSuchAlgorithmException e) {
            e.printStackTrace();
        } catch (InvalidKeySpecException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return ans;
    }


    private SecretKey generateKey(String password)
            throws NoSuchAlgorithmException, InvalidKeySpecException, UnsupportedEncodingException {
        // Number of PBKDF2 hardening rounds to use. Larger values increase
        // computation time. You should select a value that causes computation
        // to take >100ms.
        byte[] key = password.getBytes("UTF-8");
        return new SecretKeySpec(Arrays.copyOf(key,16), "AES");
    }


}
