package com.gavid.ai.service;

import java.io.IOException;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import com.gavid.ai.common.exception.HyException;
import com.gavid.ai.request.AiChatRequest;

/**  
* @Title: ChatService.java  
* @package com.gavid.ai.service
* @Description: TODO(用一句话描述该文件做什么)
* @date 2023年11月27日  
* @version V1.0  
*/
public interface IChatService {

	/**  
	* @Title: streamChatWithWeb  
	* @Description: 流式问答   
	* @param aiChatRequest
	* @param request
	* @param response
	* @throws HyException
	* @throws IOException    参数  
	* @return void    返回类型  
	*/ 
	void streamChatWithWeb(AiChatRequest aiChatRequest, HttpServletRequest request, HttpServletResponse response) throws HyException, IOException;
}
