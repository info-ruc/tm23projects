package com.gavid.ai.common.xfyun;

import java.nio.charset.StandardCharsets;
import java.security.InvalidKeyException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.SignatureException;
import java.util.Base64;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

/**  
* @Title: CryptTools.java  
* @package com.gavid.common.xfyun
* @Description: 加解密工具
* @version V1.0  
*/
public class CryptTools {
	public static final String HMAC_SHA1 = "HmacSHA1";

    public static final String HMAC_SHA256 = "HmacSHA256";

    private static final char[] md5String = {
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'a', 'b', 'c', 'd', 'e', 'f'};


	/**  
	* @Title: hmacEncrypt  
	* @Description: HMAC加解密 
	* @param encryptType
	* @param plainText
	* @param encryptKey
	* @return
	* @throws SignatureException    参数  
	* @return String    返回类型  
	*/ 
	public static String hmacEncrypt(String encryptType, String plainText, String encryptKey)
			throws SignatureException {

		try {
			byte[] data = encryptKey.getBytes(StandardCharsets.UTF_8);
			SecretKeySpec secretKey = new SecretKeySpec(data, encryptType);
			Mac mac = Mac.getInstance(encryptType);
			mac.init(secretKey);
			byte[] text = plainText.getBytes(StandardCharsets.UTF_8);
			byte[] rawHmac = mac.doFinal(text);
			return Base64.getEncoder().encodeToString(rawHmac);
		} catch (InvalidKeyException e) {
			throw new SignatureException("InvalidKeyException:" + e.getMessage());
		} catch (NoSuchAlgorithmException e) {
			throw new SignatureException("NoSuchAlgorithmException:" + e.getMessage());
		}
	}

	/**  
	* @Title: md5Encrypt  
	* @Description: Md5加密
	* @param pstr
	* @return
	* @throws SignatureException    参数  
	* @return String    返回类型  
	*/ 
	public static String md5Encrypt(String pstr) throws SignatureException {

		try {
			byte[] btInput = pstr.getBytes();
			MessageDigest mdInst = MessageDigest.getInstance("MD5");
			mdInst.update(btInput);
			byte[] md = mdInst.digest();
			int j = md.length;
			char[] str = new char[j * 2];
			int k = 0;
			for (byte byte0 : md) {
				str[k++] = md5String[byte0 >>> 4 & 0xF];
				str[k++] = md5String[byte0 & 0xF];
			}
			return new String(str);
		} catch (NoSuchAlgorithmException e) {
			throw new SignatureException("NoSuchAlgorithmException:" + e.getMessage());
		}
	}

    /**  
    * @Title: base64Encode  
    * @Description: BASE64加密
    * @param plainText
    * @return    参数  
    * @return String    返回类型  
    */ 
    public static String base64Encode(String plainText) {
        return Base64.getEncoder().encodeToString(plainText.getBytes(StandardCharsets.UTF_8));
    }
}
