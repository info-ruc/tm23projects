package com.scienceriver.esmaoj17.es8knnsearch.renda;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.elasticsearch.annotations.Document;
import org.springframework.data.elasticsearch.annotations.Field;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Document(createIndex = false,indexName = "rnda_dense_index768")
public class AIFaqEntity {

    @Field("id")
    private String id;

    @Field("doi")
    private String doi;

    @Field("article_id")
    private String article_id;

    @Field("title_cn")
    private String title_cn;

    @Field("title_original")
    private String title_original;

    @Field("faq_question_cn")
    private String faq_question_cn;

    @Field("faq_question_en")
    private String faq_question_en;

    @Field("faq_answer_cn")
    private String faq_answer_cn;
    @Field("faq_answer_en")
    private String faq_answer_en;

    @Field("faq_question_en_vector")
    private Float[] faq_question_en_vector;
    @Field("faq_question_cn_vector")
    private Float[] faq_question_cn_vector;
    private double score;
}
