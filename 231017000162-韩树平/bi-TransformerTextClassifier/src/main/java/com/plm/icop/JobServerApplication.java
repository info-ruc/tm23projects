package com.plm.icop;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.openfeign.EnableFeignClients;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.transaction.annotation.EnableTransactionManagement;

@SpringBootApplication
@EnableScheduling
@EnableFeignClients
public class JobServerApplication {

    public static void main(String[] args) {
        SpringApplication.run(JobServerApplication.class, args);
    }
}
