package com.gavid.ai.controller;

import com.gavid.ai.request.AiChatRequest;
import com.gavid.ai.service.IChatService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.ModelAndView;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.Date;

/**  
* @Title: IndexController.java
* @package com.gavid.ai.controller
* @Description: TODO(用一句话描述该文件做什么)
* @date 2023年11月27日  
* @version V1.0  
*/
@Controller
@RequestMapping("/")
public class IndexController {
	private static final Logger log = LoggerFactory.getLogger(IndexController.class);
	@Autowired
	private IChatService chatServiceImpl;
	
    @GetMapping
    public ModelAndView toPage(HttpServletRequest req, HttpServletResponse res) {
        ModelAndView mav = new ModelAndView("chat/chat");
        return mav;
    }
}
