var ResViews = {
	searchPubRes : function(resId) {
		var a = false;
		$.ajax({
			type : "POST",
			url : "/biz/pubRes/pagePubRes",
			async : false,
			data : {
				resId : resId
			},
			success : function(data) {
				if (data.list.length > 0) {
					var resType = "";
					$.ajax({
						type : "POST",
						url : "/res/getOnlyResList",
						async : false,
						data : {
						resId : resId
						},
						success : function(data) {
						resType = data.list[0].resType;
						}
					});
					a = true;
					if ("volunteer" == resType) {
						layer.msg("志愿者资源暂时不支持预览! ");
						return a;
					} else if ("document" == resType) {
						layer.msg("文档资源暂时不支持预览! ");
						return a;
					}
					var pcUrl = ResViews.getWebUrl(resId);
                    /*var localUrl = window.location.href;
                    if(localUrl.indexOf('cc.hljysg.org.cn')>0){
                    	pcUrl="http://cc.hljysg.org.cn";
                    }else{
                    	pcUrl='http://172.22.9.43/pc';
                    }*/
					window.open(pcUrl + "?resType=" + resType
							+ "&pubId=" + data.list[0].id);
				}
			}
		});
		return a;
	},
	viewRes : function(resId) {
		var a = this.searchPubRes(resId);
		if (!a) {
			var params = {
				pubCatId : '99999',
				resId : resId,//
				status : 'res_pub_state_0',
				deviceType : 'PUB_TYPE_1',
				orderNo : 1
			};
			$.ajax({
				type : "POST",
				url : "/biz/pubRes/pubResSave",
				async : false,
				data : params,
				success : function(data) {
					ResViews.searchPubRes(resId);
				}
			});
		}
	},
	getWebUrl : function(logicId) {
		var webUrl = '';
		$.ajax({
			type : "POST",
			url : "/sysDict/getWebUrl",
			async : false,
			data : {
				logicId : logicId
			},
			success : function(data) {
				webUrl = data.webUrl;
			}
		});
		return webUrl;
	},
	
	
}