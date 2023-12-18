package com.plm.icop.configuration;

import com.plm.core.com.service.ComSysConfigService;
import com.plm.icop.job.TransformerTextClassifierJob;
import org.springframework.beans.BeansException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.config.AutowireCapableBeanFactory;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationContextAware;
import org.springframework.stereotype.Component;

/**
 * Need class description here...
 *
 * @Date: 2020/1/28
 */

@Component
public class StatsApplicationContextProvider implements ApplicationContextAware {

	@Autowired
	private ComSysConfigService comSysConfigService;


	@Override
	public void setApplicationContext(ApplicationContext applicationContext) throws BeansException {
		// init job beans
		createJobBeans(applicationContext);
	}

	private void createJobBeans(ApplicationContext applicationContext) {
		// PoolCoinUpdateJob
		AutowireCapableBeanFactory factory = applicationContext.getAutowireCapableBeanFactory();


		TransformerTextClassifierJob icopReceiveMailJob = new TransformerTextClassifierJob(comSysConfigService);
		factory.autowireBean(icopReceiveMailJob);
		factory.initializeBean(icopReceiveMailJob, TransformerTextClassifierJob.class.getSimpleName());

	}
}
