package com.gavid.ai.common.xfyun.spark;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

import javax.servlet.AsyncContext;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.apache.logging.log4j.util.Strings;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.gavid.ai.common.enums.HyCodeEnum;
import com.gavid.ai.common.utils.R;
import com.gavid.ai.common.utils.ResponseUtils;
import com.gavid.ai.domain.chat.ai.ChatPrompt;
import com.gavid.ai.domain.xfyun.spark.SparkParameter;
import com.gavid.ai.request.chat.ai.ChatRequest;
import com.gavid.ai.request.xfyun.spark.SparkMessageRequest;
import com.gavid.ai.response.chat.ai.ChatResponse;
import com.gavid.ai.response.xfyun.spark.SparkMessageResponse;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.WebSocket;
import okhttp3.WebSocketListener;

/**  
* @Title: SparkChatComponent.java  
* @package com.gavid.ai.common.xfyun.spark
* @Description: TODO(用一句话描述该文件做什么)
* @version V1.0  
*/
@Component
public class SparkChatComponent {
	private static final Logger log = LoggerFactory.getLogger(SparkChatComponent.class);
	// 构建一个线程安全的 Map 对象 用于缓存response响应对象，在回调中使用
	private static final Map<String, RequestManager> requestManagerMap = new ConcurrentHashMap<>();
	// Websocket 链接管理器
	private static final Map<String, WebSocket> webSocketManager = new ConcurrentHashMap<>();
	private static Map<String, List<ChatPrompt>> cacheChatPrompt = new ConcurrentHashMap<>();
//    private static int expireTime = 0;
	
	private static final String APPID = "a82eee97";
	private static final String APPKEY = "a8fb78e48e44ea049afb81c07e5b20a8";
	private static final String APPSECRET = "NGM1N2RkYWZjZWIwMDdlZWZjMDU5NDMz";
    private static final String CHAT_HISTORY_KEY = "chat_history_";
    
	@Autowired
	private AsyncContextListener asyncContextListener;
	
	// 请求设置
    private OkHttpClient client = new OkHttpClient.Builder()
            .readTimeout(30, TimeUnit.SECONDS)//设置读取超时时间
            .writeTimeout(30, TimeUnit.SECONDS)//设置写的超时时间
            .connectTimeout(30, TimeUnit.SECONDS)//设置连接超时时间
            .pingInterval(30, TimeUnit.SECONDS)
            .build();
    
	/**  
	* @Title: getWebSocket  
	* @Description: 获取用户id 相同的 websocket 链接  
	* @param userId
	* @param chatId
	* @return    参数  
	* @return WebSocket    返回类型  
	*/ 
	private WebSocket getWebSocket(String userId, String chatId) {
		String key = userId + "_" + chatId;
		if (webSocketManager.containsKey(key)) {
			return webSocketManager.get(key);
		}
		// 获取鉴权url
		SparkAuthBuilder sparkAuth = SparkAuthBuilder.builder().apiKey(APPKEY).apiSecret(APPSECRET).build();
		// 创建webSocket链接
		Request wsRequest = new Request.Builder().url(sparkAuth.getAuthUrl()).build();
		WebSocket webSocket = client.newWebSocket(wsRequest, new SparkWebSocket(userId, chatId));
		webSocketManager.put(key, webSocket);
		return webSocket;
	}
    
    /**
     * 构造请求参数，发送请求
     *
     * @param chatRequest
     */
	public void send(ChatRequest chatRequest, HttpServletResponse response, HttpServletRequest request) {
		String userId = chatRequest.getUserId();
		String chatHistoryKey = CHAT_HISTORY_KEY + userId;
		SparkMessageRequest.Header header = new SparkMessageRequest.Header(APPID, userId);
		// 获取历史对话信息
		List<ChatPrompt> chatPromptList = new ArrayList<>(chatRequest.getMessageList());
		if (cacheChatPrompt.containsKey(chatHistoryKey)) {
			chatPromptList = cacheChatPrompt.get(chatHistoryKey);
			// 如果数量超过10条则删除一半
			if (chatPromptList.size() > 10) {
				chatPromptList = chatPromptList.subList(chatPromptList.size() / 2, chatPromptList.size());
			}
			List<ChatPrompt> messages = chatRequest.getMessageList();
			chatPromptList.addAll(messages);
		}
		// 固定参数
		SparkParameter parameter = new SparkParameter("generalv2", 0.5d, 4096d, 4d, chatRequest.getChatId());
		// 设置请求体
		SparkMessageRequest.Payload payload = new SparkMessageRequest.Payload(chatPromptList);

		SparkMessageRequest sparkMessageRequest = new SparkMessageRequest(header, payload, parameter);
		String message = JSONObject.toJSONString(sparkMessageRequest);
		log.info("spark chat message: {}", message);
		WebSocket webSocket = this.getWebSocket(chatRequest.getUserId(), chatRequest.getChatId());
		boolean sendFlag = webSocket.send(message);
		if (sendFlag) {
			// 开启异步返回 阻塞超时最多30s
			AsyncContext asyncContext = request.startAsync();
			asyncContext.setTimeout(30000L);
			asyncContext.addListener(asyncContextListener);
			response.setContentType(MediaType.TEXT_EVENT_STREAM_VALUE);
			response.setCharacterEncoding(StandardCharsets.UTF_8.toString());
			requestManagerMap.put(chatRequest.getUserId() + "_" + chatRequest.getChatId(),
					new RequestManager(asyncContext, response));
			// 缓存问题
			cacheChatPrompt.put(chatHistoryKey, chatPromptList);
		} else {
			ResponseUtils.write(response, JSONObject
					.toJSONString(R.error(HyCodeEnum.API_CALL_ERROR.getCode(), HyCodeEnum.API_CALL_ERROR.getMsg())));
		}

	}
    
    /**
     * 请求管理器 管理异步请求对象 和 响应对象
     */
	public static class RequestManager {
		AsyncContext asyncContext;

		HttpServletResponse response;
		/**
		 * 回复信息保存
		 */
		StringBuilder requestContent;

		private int asyncCloseFlag;

		public RequestManager() {
		}

		public RequestManager(AsyncContext asyncContext, HttpServletResponse response) {
			this.asyncContext = asyncContext;
			this.response = response;
			this.requestContent = new StringBuilder();
		}

		public void closeAsyncComplete() {
			if (this.asyncCloseFlag == 1) {
				return;
			}
			this.asyncCloseFlag = 1;
			this.asyncContext.complete();
		}

		public StringBuilder getRequestContent() {
			return requestContent;
		}

		public void setRequestContent(StringBuilder requestContent) {
			this.requestContent = requestContent;
		}

		public AsyncContext getAsyncContext() {
			return asyncContext;
		}

		public void setAsyncContext(AsyncContext asyncContext) {
			this.asyncContext = asyncContext;
		}

		public HttpServletResponse getResponse() {
			return response;
		}

		public void setResponse(HttpServletResponse response) {
			this.response = response;
		}
	}
	
	public class SparkWebSocket extends WebSocketListener {
		private String userId;
		private String chatId;

		private String responseKey;

		public SparkWebSocket(String userId, String chatId) {
			this.userId = userId;
			this.chatId = chatId;
			this.responseKey = userId + "_" + chatId;
		}

		public String getChatId() {
			return chatId;
		}

		public void setChatId(String chatId) {
			this.chatId = chatId;
		}

		public String getUserId() {
			return userId;
		}

		public void setUserId(String userId) {
			this.userId = userId;
		}

		@Override
		public void onMessage(@NotNull WebSocket webSocket, @NotNull String text) {
			log.info("spark chat message: {}", text);
			super.onMessage(webSocket, text);
			// 将消息返回给客户端
			SparkMessageResponse response = JSON.parseObject(text, SparkMessageResponse.class);
			// 获取Response 响应对象
			RequestManager manager = requestManagerMap.get(this.responseKey);
			if (null == manager || null == manager.getResponse()) {
				return;
			}
			// 状态为2时表示回复完毕
			boolean isClose = response.getHeader().getStatus() == 2;
			// 统一响应类
			List<ChatPrompt> respTexts = response.getPayload().getChoices().getText();
			ChatResponse chatResponse = new ChatResponse();
			chatResponse.setMessageList(respTexts);
			chatResponse.setId(response.getHeader().getSid());
			// 保存回复信息
			String returnContent = chatResponse.getMessageList().get(0).getContent();
			if (Strings.isNotBlank(returnContent)) {
				StringBuilder requestContent = manager.getRequestContent();
				requestContent.append(returnContent);
			}
			// 响应数据流
			respTexts.stream().forEach(t->{
				ResponseUtils.writeStream(manager.getAsyncContext().getResponse(), t.getContent(), isClose);
			});
			

			if (isClose) {
				// 获取header
				if (response.getHeader().getCode() != 0) {
					// 会话结束
					webSocket.close(1000, "会话结束");
				}

				// 关闭异步响应 防止线程阻塞
				manager.closeAsyncComplete();
				// 缓存回复到历史记录中
				String chatHistoryKey = CHAT_HISTORY_KEY + this.userId;
				List<ChatPrompt> chatPromptList = new ArrayList<>();
				if(cacheChatPrompt.containsKey(chatHistoryKey)) {
					chatPromptList = cacheChatPrompt.get(chatHistoryKey);
				}
//				if (Strings.isNotBlank(messageListStr)) {
//					chatPromptList = JSON.parseArray(messageListStr, ChatPrompt.class);
//				}
				chatPromptList.add(new ChatPrompt(manager.getRequestContent().toString(), "assistant"));
				cacheChatPrompt.put(chatHistoryKey, chatPromptList);
//				redisUtil.set(RedisKeyConstant.REDIS_CHAT_HISTORY_KEY + this.userId, JSON.toJSONString(chatPromptList),
//						expireTime);

			}

		}

		@Override
		public void onOpen(@NotNull WebSocket webSocket, @NotNull Response response) {
			log.info("spark ws connection response: {}", response.code());
			super.onOpen(webSocket, response);
		}

		@Override
		public void onFailure(@NotNull WebSocket webSocket, @NotNull Throwable t, @Nullable Response response) {
			super.onFailure(webSocket, t, response);
			log.error("onFailure code:" + t.getMessage());
			try {
				// WebSocket响应失败后不会进入onClosed方法，需要手动关闭
				if (!webSocket.close(1001, "on failure")) {
					webSocket.cancel();
				}
				webSocketManager.remove(this.responseKey);

				if (null != response) {
					int code = response.code();
					log.error("onFailure code:" + code);
					log.error("onFailure body:" + response.body().string());
					// 101 为连接失败
					if (101 != code) {
						log.error("spark ws connection failed");
					}
				}
			} catch (IOException e) {
				log.error("onFailure :" + e.getMessage());
			}

			ResponseUtils.write(requestManagerMap.get(this.responseKey).getResponse(),
					JSONObject.toJSONString(R.error(HyCodeEnum.ERROR.getCode(), HyCodeEnum.ERROR.getMsg())));
		}

		@Override
		public void onClosed(@NotNull WebSocket webSocket, int code, @NotNull String reason) {
			super.onClosed(webSocket, code, reason);
			log.info("spark ws connection closed and code: {},{}", reason, code);
		}

		@Override
		public void onClosing(@NotNull WebSocket webSocket, int code, @NotNull String reason) {
			super.onClosing(webSocket, code, reason);
			// 连接被关闭时， 删除缓存的websocket对象
			if (!webSocket.close(code, reason)) {
				webSocket.cancel();
			}
			webSocketManager.remove(this.responseKey);
		}

	}

}
