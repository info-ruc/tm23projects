package com.plm.icop.config;

import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import javax.annotation.Resource;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Import;
import org.springframework.scheduling.annotation.EnableScheduling;

import com.plm.common.service.RetrofitTemplateService;
import com.plm.core.com.service.ComSysConfigService;
import com.plm.core.config.CoreConfiguration;


@Configuration
@EnableScheduling
@Import({CoreConfiguration.class})
@ComponentScan(basePackages = "com.plm.icop")
public class ServerConfiguration {

    @Resource
    private ComSysConfigService sysConfigService;

    @Bean("feePercentUpdateExecutor")
    public Executor feePercentUpdateExecutor() {
        return Executors.newSingleThreadExecutor();
    }

    @Bean
    public RetrofitTemplateService retrofitTemplateService() {
        return new RetrofitTemplateService();
    }
}
