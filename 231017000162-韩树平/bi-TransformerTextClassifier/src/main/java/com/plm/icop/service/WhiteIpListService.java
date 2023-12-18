package com.plm.icop.service;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import javax.annotation.PostConstruct;
import javax.annotation.Resource;

import org.apache.commons.lang3.StringUtils;
import org.springframework.stereotype.Service;

import com.plm.core.com.service.ComSysConfigService;

import lombok.extern.slf4j.Slf4j;

@Slf4j
@Service
public class WhiteIpListService {

    private final Map<String, Boolean> nodeIpSet = new ConcurrentHashMap<>();

    @Resource
    private ComSysConfigService sysConfigService;

    public boolean isAllowAll() {
        return nodeIpSet.size() == 1 && nodeIpSet.containsKey("all");
    }

    public boolean checkNodeIp(String nodeIp) {
        if (StringUtils.isBlank(nodeIp)) {
            return false;
        }
        if (nodeIpSet.size() == 0) {
            return false;
        }
        return nodeIpSet.containsKey(nodeIp);
    }

    @PostConstruct
    public void refresh() {
    }
}
