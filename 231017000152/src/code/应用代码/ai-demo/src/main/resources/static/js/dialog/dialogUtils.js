var DialogUtils ={
		openDialog:function(title, url, isEndFunction) {
			var index = layer.open({
		        type: 2,
		        title: title,
		        maxmin: true,
		        shadeClose: false, // 点击遮罩关闭层
		        area: ['1000px', '620px'],
		        content: url,
		        end: function () {
		        	if(isEndFunction){
		        		location.reload();
			        }
	            }
		    });
			layer.full(index);
		},
		/**
		 * selectType 单选(radio)、多选(checkbox)
		 * selectorId 编号隐藏文本域id
		 * selectorName 名称隐藏文本域id
		 * */
		openUserForm:function(selectType, selectorId, selectorName) {
			var index = layer.open({
		        type: 2,
		        title: '选择用户',
		        maxmin: true,
		        shadeClose: false, // 点击遮罩关闭层
		        area: ['1200px', '520px'],
		        btn: ['确定', '关闭'],
		        content: '/user/userDialog?selectType='+selectType,
		        yes: function(index, layero){ //或者使用btn1
		        	var iframeWin = window[layero.find('iframe')[0]['name']];
		        	var selectUsers = iframeWin.getCheckData();
		        	var userIds ='', userNames=''; 
		        	for(var i = 0; i < selectUsers.length;i++){
		        		var user = selectUsers[i];
		        		userIds+=user.id;
		        		userIds+=',';
		        		userNames+=user.name;
		        		userNames+=',';
		        	}
		        	userIds = userIds.slice(0,userIds.length-1);
		        	$('#'+selectorName).val(userNames);
		        	$('#'+selectorId).val(userIds);
		            layer.close(index);
		        },cancel: function(index){ //或者使用btn2
		 
		        }
		    });
		},
		/**
		 * selectType 单选(radio)、多选(checkbox)
		 * selectorId 编号隐藏文本域id
		 * selectorName 名称隐藏文本域id
		 * */
		openDeptForm:function(selectType, selectorId, selectorName) {
			var index = layer.open({
		        type: 2,
		        title: '选择部门',
		        maxmin: true,
		        shadeClose: false, // 点击遮罩关闭层
		        area: ['1200px', '520px'],
		        btn: ['确定', '关闭'],
		        content: '/sys/department/deptDialog?selectType='+selectType,
		        yes: function(index, layero){ //或者使用btn1
		        	var iframeWin = window[layero.find('iframe')[0]['name']];
		        	var selectDepts = iframeWin.getCheckData();
		        	var deptIds ='', deptNames=''; 
		        	for(var i = 0; i < selectDepts.length;i++){
		        		var dept = selectDepts[i];
		        		deptIds+=dept.id;
		        		deptIds+=',';
		        		deptNames+=dept.name;
		        		deptNames+=',';
		        	}
		        	deptIds = deptIds.slice(0,deptIds.length-1);
		        	$('#'+selectorName).val(deptNames);
		        	$('#'+selectorId).val(deptIds);
		            layer.close(index);
		        },cancel: function(index){ //或者使用btn2
		 
		        }
		    });
		},
		openSeatTemplateList:function(selectType, selectorId, selectorName) {
			var index = layer.open({
		        type: 2,
		        title: '选择座位模板',
		        maxmin: true,
		        shadeClose: false, // 点击遮罩关闭层
		        area: ['1200px', '520px'],
		        btn: ['确定', '关闭'],
		        content: '/sys/seat/template?selectType='+selectType,
		        yes: function(index, layero){ //或者使用btn1
		        	var iframeWin = window[layero.find('iframe')[0]['name']];
		        	var seatTemplates = iframeWin.getCheckData();
		        	var templateIds ='', templateNames=''; 
		        	for(var i = 0; i < seatTemplates.length;i++){
		        		var seatTemplate = seatTemplates[i];
		        		templateIds+=seatTemplate.id;
		        		templateIds+=',';
		        		templateNames+=seatTemplate.templateName;
		        		templateNames+=',';
		        	}
		        	templateIds = templateIds.slice(0,templateIds.length-1);
		        	$('#'+selectorName).val(templateNames);
		        	$('#'+selectorId).val(templateIds);
		            layer.close(index);
		        },cancel: function(index){ //或者使用btn2
		 
		        }
		    });
		},
		deploymentProcDialog:function() {
			var index = layer.open({
		        type: 2,
		        title: '选择已部署流程',
		        maxmin: true,
		        shadeClose: false, // 点击遮罩关闭层
		        area: ['1200px', '520px'],
		        btn: ['确定', '关闭'],
		        content: '/activiti/process/toDeploymentProcDialog',
		        yes: function(index, layero){ //或者使用btn1
		        	var iframeWin = window[layero.find('iframe')[0]['name']];
		        	var selectDeployment = iframeWin.getCheckData()[0];
		        	$('input[name=deploymentId]').val(selectDeployment.deploymentId);
		        	$('input[name=procName]').val(selectDeployment.name);
		        	var id = selectDeployment.id;
		        	id+="";
		        	$('input[name=procKey]').val(id.split(":")[0]);
		            layer.close(index);
		        },cancel: function(index){ //或者使用btn2
		 
		        }
		    });
		},
		nextStepForm:function() {
			var index = layer.open({
		        type: 2,
		        title: '选择下一步',
		        maxmin: true,
		        shadeClose: false, // 点击遮罩关闭层
		        area: ['1500px', '700px'],
		        content: '/activiti/process/nextStepDialog'
		    });
		},
		taskExecutionProc:function(procInsId, oldProcInsIds) {
		    layer.open({
		        type: 2,
		        title: '流程执行过程',
		        maxmin: true,
		        shadeClose: false, // 点击遮罩关闭层
		        area: ['1200px', '520px'],
		        content: '/activiti/task/toTaskExecutionProc?procInsId=' + procInsId+'&oldProcInsIds='+oldProcInsIds
		    });
		},
		openUploadForm:function(resIds, editType) {
			var index = layer.open({
		        type: 2,
		        title: '附件上传',
		        maxmin: true,
		        shadeClose: false, // 点击遮罩关闭层
		        area: ['1000px', '520px'],
		        btn: ['确定', '关闭'],
		        content: '/upload/toUpload/?resIds='+resIds+'&editType='+editType,
		        yes: function(index, layero){ //或者使用btn1
		        	var iframeWin = window[layero.find('iframe')[0]['name']];
			        var ossPath = iframeWin.getUploadData();
			        var attachmentNames = iframeWin.getAttachmentNames();
			        var delResIds = iframeWin.getDelResIds();
			        var attachmentSizes = iframeWin.getAttachmentSizes();
			        $("#delResIds").val(delResIds);
			        $("#attachmentUrls").val(ossPath);
			        $("#attachmentNames").val(attachmentNames);
			        $("#attachmentSizes").val(attachmentSizes);
		            layer.close(index);
		        },cancel: function(index){ //或者使用btn2
		 
		        }
		    });
		},
		openProjectForm:function(selectType, selectorId, selectorName, selectorContractNo, selectorMarketer, queryBatch) {
			var index = layer.open({
		        type: 2,
		        title: '选择项目',
		        maxmin: true,
		        shadeClose: false, // 点击遮罩关闭层
		        area: ['1200px', '520px'],
		        btn: ['确定', '关闭'],
		        content: '/inv/project/toList?selectType='+selectType,
		        yes: function(index, layero){ //或者使用btn1
		        	var iframeWin = window[layero.find('iframe')[0]['name']];
		        	var selectProjects = iframeWin.getCheckData();
		        	if(selectProjects && selectProjects.length > 0){
		        		$('#'+selectorName).val(selectProjects[0].projectName);
			        	$('#'+selectorId).val(selectProjects[0].id);
			        	$('#'+selectorContractNo).val(selectProjects[0].contractNo);
			        	$('#'+selectorMarketer).val(selectProjects[0].marketer);
			        	if(queryBatch){//带出批次
			        		$.ajax({
			        			type : "post",
			        			dataType : "json",
			        			async:true,
			        			url : "/inv/project/batch/findByProjectId/"+selectProjects[0].id,
			        			success : function(data) {
			        				$("#projectBatchId").html('<option value="">请选择批次</option>');
			        				if(data.code == 0 && data.data){
			        					var html = "";
			        					$(data.data).each(function(i,obj){
			        						html+='<option value="'+obj.id+'">'+obj.batch+'</option>';
			        					});
			        					$("#projectBatchId").append(html);
			        					form.render("select");
			        				}
			        			}
			        		});
			        	}
		        	}else{
		        		
		        	}
		        	
		            layer.close(index);
		        },cancel: function(index){ //或者使用btn2
		 
		        }
		    });
		},
		openMachineForm:function(selectType, selector, field, machineStatus, projectId, projectBatchId) {
			let url = '/inv/machine/toList?selectType='+selectType+'&machineStatus='+machineStatus;
			if(projectId != null && projectId != ''){
				url+='&projectId='+projectId;
			}
			if(projectBatchId != null && projectBatchId != ''){
				url+='&projectBatchId='+projectBatchId;
			}
			var index = layer.open({
		        type: 2,
		        title: '选择设备',
		        maxmin: true,
		        shadeClose: false, // 点击遮罩关闭层
		        area: ['1200px', '520px'],
		        btn: ['确定', '关闭'],
		        content: url,
		        yes: function(index, layero){ //或者使用btn1
		        	var iframeWin = window[layero.find('iframe')[0]['name']];
		        	if(field == 'deliveryNum'){
		        		var selectMachines = iframeWin.getCheckData();
			        	if(selectMachines && selectMachines.length > 0){
			        		$(selector).val(selectMachines.length);
			        		var table = deliveryDetailTable;
			        		var oldData = table.cache.detailData;
			        		table.reload('detailData',{data:[]});//先清空数据
			        		var machineArr = selectMachines.concat(oldData);
			        		var temp = {}, result=[];
			        		machineArr.map((item,index)=>{
			        	         if (!temp[item.id]){
			        	             result.push(item);
			        	             temp[item.id] =  true
			        	         }
			        	     })
			        		table.reload('detailData',{data:result}); 
			        	}else{
			        		
			        	}
		        	}else if(field == 'machineSn'){
		        		var selectMachines = iframeWin.getCheckData();
			        	if(selectMachines && selectMachines.length > 0){
			        		$(selector).val(selectMachines[0].machineSn);
			        	}
		        	}
		            layer.close(index);
		        },cancel: function(index){ //或者使用btn2
		 
		        }
		    });
		}

}