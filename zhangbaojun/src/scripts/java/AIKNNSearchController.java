package com.scienceriver.esmaoj17.es8knnsearch.renda;

import com.scienceriver.esmaoj17.wordssplit.jieba.JavaJiebaSegmenter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.*;

import static java.util.Map.Entry.comparingByValue;
import static java.util.stream.Collectors.toMap;

@Slf4j
@RestController
public class AIKNNSearchController {
	@Autowired
	private AIKNNSearchService knnSearchService;
	private JavaJiebaSegmenter jiebaSegmenter=JavaJiebaSegmenter.getInstance();
	@GetMapping("/knnsearchCN")
	public List<AIFaqEntity> knnSearch(@RequestParam("sentence") String sentence) {
		/**汉语短句需要关键词提取和增强，否则语义理解可能与人类不一致***/
		String enhancedSentence=jiebaSegmenter.enhanceKeyWord(sentence);

		Map<String,String> vecStr=knnSearchService.embeddingSentence(enhancedSentence);
		return knnSearchService.knnSearch(vecStr,"CN");
	}

	@GetMapping("/knnsearchEN")
	public List<AIFaqEntity> knnSearchEN(@RequestParam("sentence") String sentence) {
		/**英语不需要关键词提取和增强***/
		Map<String,String> vecStr=knnSearchService.embeddingSentence(sentence);
		return knnSearchService.knnSearch(vecStr,"EN");
	}

//	@GetMapping("/embeddingSentence")
//	public Map<String,String> embeddingSentence(@RequestParam("sentence") String sentence){
//		System.out.println("embeddingSentenceController:"+sentence);
//		return knnSearchService.embeddingSentence(sentence);
//	}
}
