package com.scienceriver.esmaoj17.es8knnsearch.renda;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch._types.KnnQuery;
import co.elastic.clients.elasticsearch.core.SearchResponse;
import co.elastic.clients.elasticsearch.core.search.Hit;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Slf4j
@Service
public class AIKNNSearchService {
	@Autowired
	private ElasticsearchClient elasticsearchClient;
	private WebClient webClient=null;

	private static ObjectMapper objectMapper=new ObjectMapper();
	@Value("${knn.index:renda_dense_index768}")
	private String INDEX_NAME;
	@Value("${knn.port:8002}")
	private int knnApiPort;
	@Value("${knn.host:10.230.107.101}")
	private String knnApiHost;
	public AIKNNSearchService(){
		webClient= WebClient.builder().baseUrl("")
//				.defaultHeaders()
//				.defaultCookies()
				.build();
	}
	public List<AIFaqEntity> knnSearch(Map<String,String> vec,String language) {
		try {
//			String vecStr=objectMapper.writeValueAsString(vec.get("vec"));
			String vecStr=vec.get("vec");
			System.out.println("vecStr:"+vecStr);

			//把Entity转换成数组
			List<Float> vectors=objectMapper.readValue(vecStr, new TypeReference<List<Float>>() {});

			KnnQuery knnQuery = KnnQuery.of(m ->
					m.field(language.equals("EN")? "faq_question_en_vector":"faq_question_cn_vector")
							.queryVector(vectors)
							.k(10)
							.numCandidates(100)
			);

//			String str="感冒症状";
			/**向量与普通关键词查询聚合结果,本次暂时不用****/
//			Query query = Query.of(ss ->
//					ss.match(hh->hh.field(language.equals("EN")? "faq_question_en":"faq_question_cn").query(q->q.stringValue(str))));

			SearchResponse<AIFaqEntity>
					searchResult = elasticsearchClient.search(s ->
							s.index(INDEX_NAME)
//							.query(query)
							.knn(knnQuery)
					, AIFaqEntity.class
			);

			List<AIFaqEntity> result = new ArrayList<>();
			for (Hit<AIFaqEntity> hit : searchResult.hits().hits()) {
				AIFaqEntity entitySource = hit.source();
				entitySource.setScore(hit.score());
				entitySource.setFaq_question_cn_vector(null);
				entitySource.setFaq_question_en_vector(null);
				result.add(entitySource);
			}
			log.info("SearchResponse=\n{}", objectMapper.writeValueAsString(result));
			return result;
		}catch (IOException e){
			e.printStackTrace();
			return null;
		}
	}

	public Map<String,String> embeddingSentence(String sentence){
		System.out.println("embeddingSentenceService:"+sentence);
		Mono<Map> result=webClient.get()
				.uri(builder->
						builder.scheme("http")
								.host(knnApiHost)
								.port(knnApiPort)
								.path("/embeddingsentence")
								.queryParam("sentence",sentence)
								.build()
				)
				.retrieve().bodyToMono(Map.class);
		return result.block();
	}
}
