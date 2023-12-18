package com.plm.icop.utils;

import com.plm.common.constant.CoinType;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;

@Slf4j
public class CoinTypeUtil {

    public static CoinType getCoinType(String name) {
        try {
            return StringUtils.isEmpty(name) ? null : CoinType.valueOf(name.toUpperCase());
        } catch (Exception e) {
            log.error("CoinTypeUtil getCoinType error {}", name);
            return null;
        }
    }
}
