package com.gavid.ai.controller;

import java.io.IOException;
import java.util.Date;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.servlet.ModelAndView;

import com.gavid.ai.request.AiChatRequest;
import com.gavid.ai.service.IChatService;

/**  
* @Title: ChatController.java  
* @package com.gavid.ai.controller
* @Description: TODO(用一句话描述该文件做什么)
* @date 2023年11月27日  
* @version V1.0  
*/
@Controller
@RequestMapping("chat")
public class ChatController {
	private static final Logger log = LoggerFactory.getLogger(ChatController.class);
	@Autowired
	private IChatService chatServiceImpl;
	
    @GetMapping
    public ModelAndView toPage(HttpServletRequest req, HttpServletResponse res) {
        ModelAndView mav = new ModelAndView("chat/chat");
        return mav;
    }
    
    
	@PostMapping("/streamChatWithWeb")
    @ResponseBody
	public void streamChatWithWeb(@RequestBody AiChatRequest aiChatRequest, HttpServletRequest request, HttpServletResponse response)
			throws IOException, InterruptedException {
		// 需要指定response的ContentType为流式输出，且字符编码为UTF-8
		response.setContentType("text/event-stream");
		response.setCharacterEncoding("UTF-8");
		// 禁用缓存
		response.setHeader("Cache-Control", "no-cache");
		String userName = "admin";
		String businessId = new Date().getTime()+"";
		chatServiceImpl.streamChatWithWeb(aiChatRequest, request, response);
//		return R.ok("WEB加载完毕!");
	}
    
}
