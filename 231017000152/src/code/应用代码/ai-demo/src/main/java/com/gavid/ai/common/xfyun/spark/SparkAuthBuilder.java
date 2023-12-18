package com.gavid.ai.common.xfyun.spark;

import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.security.SignatureException;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Base64;
import java.util.Locale;

import org.apache.hc.core5.net.URIBuilder;

import com.gavid.ai.common.xfyun.CryptTools;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Builder
@Data
@AllArgsConstructor
@NoArgsConstructor
public class SparkAuthBuilder {

    private static final String HOST_URL = "https://spark-api.xf-yun.com/v3.1/chat";

    private String apiKey;

    private String apiSecret;
    private static String PRE_STR = "host: %s\n" +
            "date: %s\n" +
            "GET %s HTTP/1.1";
    private static String AUTH_STR = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"";

    /**
     * 星火模型获取鉴权url
     *
     * @return
     */
    public String getAuthUrl() {
        URL url = null;
        try {
            url = new URL(HOST_URL);
            // 获取date
            String date = getGMTDate();
            // authorization 参数生成
            String preAuthStr = String.format(PRE_STR, url.getHost(), date, url.getPath());
            // SHA256加密
            String sign = CryptTools.hmacEncrypt(CryptTools.HMAC_SHA256, preAuthStr, this.getApiSecret());
            // 拼接 authorization 字符串
            String authorization = String.format(AUTH_STR, apiKey, "hmac-sha256", "host date request-line", sign);

            URI uri = new URIBuilder("https://" + url.getHost() + url.getPath()).addParameter("authorization", Base64.getEncoder().encodeToString(authorization.getBytes(StandardCharsets.UTF_8))).//
                    addParameter("date", date).//
                    addParameter("host", url.getHost()).//
                    build();
            return uri.toString().replace("http://", "ws://").replace("https://", "wss://");
        } catch (SignatureException e) {
            throw new RuntimeException(e);
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        } catch (MalformedURLException e) {
            throw new RuntimeException(e);
        }
    }

    /**  
    * @Title: getGMTDate  
    * @Description: 获取GMT格式的时间  
    * @return    参数  
    * @return String    返回类型  
    */ 
    private String getGMTDate() {
        DateTimeFormatter fmt = DateTimeFormatter.ofPattern("EEE, dd MMM yyyy HH:mm:ss z")
                .withLocale(Locale.US)
                .withZone(ZoneId.of("GMT"));

        return ZonedDateTime.now().format(fmt);
    }


}
