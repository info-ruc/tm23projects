package com.gavid.ai.common.utils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

import javax.servlet.ServletResponse;
import javax.servlet.http.HttpServletResponse;

import org.springframework.http.MediaType;

public class ResponseUtils {

	/**
     * 使用response 对象返回信息
     *
     * @param response    对象
     * @param text        需要返回的文本信息
     * @param contentType contentType 响应类型
     * @param encoding    字符编码
     */
    public static void write(HttpServletResponse response, String text, String contentType, String encoding, boolean isClose) {
        if (response == null) {
            return;
        }
        response.setContentType(contentType);
        response.setCharacterEncoding(encoding);
        try {
            OutputStream outputStream = response.getOutputStream();
            //判断 outputStream 是否关闭

            outputStream.write(text.getBytes(StandardCharsets.UTF_8));
            outputStream.flush();
            if (isClose) {
                outputStream.close();
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void write(HttpServletResponse response, String text) {
        write(response, text, MediaType.APPLICATION_JSON_VALUE, StandardCharsets.UTF_8.toString(), true);
    }

    /**
     * 返回流数据 不支持文件
     *
     * @param response       响应对象
     * @param bufferedReader 响应流
     * @param contentType    响应类型
     * @param encoding字符编码
     */
    public static void writeStream(HttpServletResponse response, BufferedReader bufferedReader, String contentType, String encoding) {
        try {
            response.setContentType(contentType);
            response.setCharacterEncoding(encoding);
            OutputStream outputStream = response.getOutputStream();
            String line = null;
            while ((line = bufferedReader.readLine()) != null) {
                // 添加换行符，在客户端获取流的时候更好解析
                line += "\n";
                outputStream.write(line.getBytes(StandardCharsets.UTF_8));
            }
            outputStream.flush();
            outputStream.close();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                bufferedReader.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * 响应 SSE(text/event-stream) 格式的推流
     *
     * @param response       响应对象
     * @param bufferedReader 响应流
     */
    public static void writeEventStream(HttpServletResponse response, BufferedReader bufferedReader) {
        writeStream(response, bufferedReader, MediaType.TEXT_EVENT_STREAM_VALUE, StandardCharsets.UTF_8.toString());
    }


    public static void write(HttpServletResponse response, String text, boolean isClose) {
        write(response, text, MediaType.APPLICATION_JSON_VALUE, StandardCharsets.UTF_8.toString(), isClose);
    }

    /**
     * 返回流数据 不支持文件
     *
     * @param response    响应对象
     * @param text        响应数字
     * @param contentType 响应类型
     * @param encoding    字符编码
     * @param isClose     是否关闭流
     */
    public static void writeStream(ServletResponse response, String text, boolean isClose) {
        try {
            response.setContentType(MediaType.TEXT_EVENT_STREAM_VALUE);
            response.setCharacterEncoding(StandardCharsets.UTF_8.toString());
            OutputStream outputStream = response.getOutputStream();
            // 需要封装成 R 格式 的统一对象
            text += "\n";
            outputStream.write(text.getBytes(StandardCharsets.UTF_8));
            outputStream.flush();
            if (isClose) {
                outputStream.close();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
