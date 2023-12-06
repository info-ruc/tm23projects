package com.scienceriver.esmaoj17.llama2.qastream;

import jakarta.servlet.AsyncContext;
import jakarta.servlet.ServletException;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.annotation.Value;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * @author barry
 * 通过java调用python大模型接口，获取答案
 * input: messages, 问题及问题上下文 通过www-form-urlencoded方式传递
 */
@WebServlet(asyncSupported = true, urlPatterns = {"/apps/intelligentqa/api/qa/v1/stream"})
public class StreamingPythonController extends HttpServlet {
//	private final static String API_URI="http://10.230.107.102:8006/chat/streamingapi";
	@Value("${api.uri:http://10.230.107.102:8007/chat/streamingapi}")
	private String API_URI;
	private final static  ExecutorService executor = Executors.newCachedThreadPool();
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		String messages=request.getParameter("messages");
		System.out.println("messages:"+messages);
// 创建异步上下文
		AsyncContext asyncContext = request.startAsync(); // 提交任务到线程池
		asyncContext.setTimeout(900000000);
		executor.submit(() -> { try {
// 请求OpenAI接口获取stream数据
			String questionParameter="messages="+ URLEncoder.encode(messages,"utf-8");
			URL url = new URL(API_URI);
			HttpURLConnection conn = (HttpURLConnection) url.openConnection();
			conn.setConnectTimeout(12000);
			conn.setReadTimeout(12000);
			conn.setRequestMethod("POST");
			conn.setDoOutput(true);
			conn.setDoInput(true);
//			conn.setRequestProperty("Content-Type", "application/json");
//			conn.setRequestProperty("Authorization", "Bearer EMPTY"); conn.setDoOutput(true);
// 			out.write("{\"prompt\": \"Hello, world!\", \"max_tokens\": 5, \"temperature\": 0.7}".getBytes());

			DataOutputStream dos = new DataOutputStream(conn.getOutputStream());
			dos.writeBytes(questionParameter);
			dos.flush();
			dos.close();

			InputStream in = conn.getInputStream();
			response.setContentType("text/event-stream");
			response.setHeader("Cache-Control", "no-cache");
			response.setHeader("Connection", "keep-alive");
			response.setHeader("Access-Control-Allow-Origin", "*");
			byte[] buffer = new byte[1024];
			int count;
//			String answer=new String();
//			int off=0;
//			int len=0;
//			while ((count = in.read(buffer,off,4)) != -1) {
// 将stream数据转发给H5页面端
//				OutputStream outputStream = asyncContext.getResponse().getOutputStream();
//				len=off+count;
//				off=len;
//				String data = new String(buffer, 0, len);//count);
//				answer+=data;
//				outputStream.write(("data: " + data + "\n\n").getBytes("utf-8"));
//				outputStream.write(("answer: " + answer + "\n\n").getBytes("utf-8"));
//				outputStream.flush();

			OutputStream outputStream =null;
			while ((count = in.read(buffer)) != -1){
				outputStream = asyncContext.getResponse().getOutputStream();
				String data = new String(buffer, 0, count);
				System.out.println("data="+data);

//				System.out.println(data.replace("data:",""));

				outputStream.write(data.getBytes("utf-8"));
				outputStream.flush();
//				try {
//					Thread.sleep(2);
//				}catch (InterruptedException e){
//					e.printStackTrace();
//				}

			}
//			System.out.println("status code:"+conn.getResponseCode());
			outputStream = asyncContext.getResponse().getOutputStream();
			outputStream.write("\n###@@@\n".getBytes("utf-8"));
			outputStream.flush();
//			outputStream.close();
			in.close();
		}
		catch (IOException e) {
			e.printStackTrace();
		}catch (Exception e){
			e.printStackTrace();
		}
		finally {
			asyncContext.complete();
		}
		});
	}
	@Override
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		doPost(request,response);
	}

	public  static void main(String[] args){
	}
}
