Date.prototype.format = function(format) {
    var date = {
           "M+": this.getMonth() + 1,
           "d+": this.getDate(),
           "h+": this.getHours(),
           "m+": this.getMinutes(),
           "s+": this.getSeconds(),
           "q+": Math.floor((this.getMonth() + 3) / 3),
           "S+": this.getMilliseconds()
    };
    if (/(y+)/i.test(format)) {
           format = format.replace(RegExp.$1, (this.getFullYear() + '').substr(4 - RegExp.$1.length));
    }
    for (var k in date) {
           if (new RegExp("(" + k + ")").test(format)) {
                  format = format.replace(RegExp.$1, RegExp.$1.length == 1
                         ? date[k] : ("00" + date[k]).substr(("" + date[k]).length));
           }
    }
    return format;
};
String.prototype.beautySub = function (len) {
    var reg = /[\u4e00-\u9fa5]/g,    //专业匹配中文
        slice = this.substring(0, len),
        chineseCharNum = (~~(slice.match(reg) && slice.match(reg).length)),
        realen = slice.length*2 - chineseCharNum;
    return this.substr(0, realen) + (realen < this.length ? "..." : "");
};
String.prototype.startWith=function(str){
	if(str==null||str==""||this.length==0||str.length>this.length)
	  return false;
	if(this.substr(0,str.length)==str)
	  return true;
	else
	  return false;
	return true;
}
String.prototype.endWith=function(str){
	if(str==null||str==""||this.length==0||str.length>this.length)
	  return false;
	if(this.substring(this.length-str.length)==str)
	  return true;
	else
	  return false;
	return true;
}
/**
* 转换xml为对象形式
* @return {Object}
* @param {XMLHttpRequest} elXML
*/
$.fn.toObject = function (){
   if (this==null) return null;
   var retObj = new Object;
   buildObjectNode(retObj,/*jQuery*/this.get(0));
   return $(retObj);
   function buildObjectNode(cycleOBJ,/*Element*/elNode){
       /*NamedNodeMap*/
       var nodeAttr=elNode.attributes;
       if(nodeAttr != null){
           if (nodeAttr.length&&cycleOBJ==null) cycleOBJ=new Object; 
           for(var i=0;i<nodeAttr.length;i++){
               cycleOBJ[nodeAttr[i].name]=nodeAttr[i].value;
           }
       }
       var nodeText="text";
       if (elNode.text==null) nodeText="textContent";
       /*NodeList*/
       var nodeChilds=elNode.childNodes;
       if(nodeChilds!=null){
           if (nodeChilds.length&&cycleOBJ==null) cycleOBJ=new Object; 
           for(var i=0;i<nodeChilds.length;i++){
               if (nodeChilds[i].tagName!=null){
                   if (nodeChilds[i].childNodes[0]!=null&&nodeChilds[i].childNodes.length<=1&&(nodeChilds[i].childNodes[0].nodeType==3||nodeChilds[i].childNodes[0].nodeType==4)){
                       if (cycleOBJ[nodeChilds[i].tagName]==null){
                           cycleOBJ[nodeChilds[i].tagName]=nodeChilds[i][nodeText];
                       }else{
                           if (typeof(cycleOBJ[nodeChilds[i].tagName])=="object"&&cycleOBJ[nodeChilds[i].tagName].length){
                               cycleOBJ[nodeChilds[i].tagName][cycleOBJ[nodeChilds[i].tagName].length]=nodeChilds[i][nodeText];
                           }else{
                               cycleOBJ[nodeChilds[i].tagName]=[cycleOBJ[nodeChilds[i].tagName]];
                               cycleOBJ[nodeChilds[i].tagName][1]=nodeChilds[i][nodeText];
                           }
                       }
                   }else{
                       if (nodeChilds[i].childNodes.length){
                           if (cycleOBJ[nodeChilds[i].tagName]==null){
                               cycleOBJ[nodeChilds[i].tagName]=new Object;
                               buildObjectNode(cycleOBJ[nodeChilds[i].tagName],nodeChilds[i]);
                           }else{
                               if (cycleOBJ[nodeChilds[i].tagName].length){
                                   cycleOBJ[nodeChilds[i].tagName][cycleOBJ[nodeChilds[i].tagName].length]=new Object;
                                   buildObjectNode(cycleOBJ[nodeChilds[i].tagName][cycleOBJ[nodeChilds[i].tagName].length-1],nodeChilds[i]);
                               }else{
                                   cycleOBJ[nodeChilds[i].tagName]=[cycleOBJ[nodeChilds[i].tagName]];
                                   cycleOBJ[nodeChilds[i].tagName][1]=new Object;
                                   buildObjectNode(cycleOBJ[nodeChilds[i].tagName][1],nodeChilds[i]);
                               }
                           }
                       }else{
                           cycleOBJ[nodeChilds[i].tagName]=nodeChilds[i][nodeText];
                       }
                   }
               }
           }
       }
   }
};
encodePwd=function(pwdStr){
	var base64 = new Base64();
	var pwd = decodeURI(pwdStr);
	return base64.encode(pwd);
};
encrypt = function(str){
	var base64 = new Base64();
	var encrypt = new JSEncrypt();
	encrypt.setPublicKey(base64.public_key());
	return encrypt.encrypt(str);
}
HTMLEnCode = function(str){
	var s = "";
	if (str.length == 0) return "";
	s = str.replace(/&/g, "&gt;");
	s = s.replace(/</g, "");
	s = s.replace(/>/g, "");
	s = s.replace(/ /g, "");
	s = s.replace(/\"/g, "");
	s = s.replace(/\'/g, "");
	s = s.replace(/\n/g, "");
	s = s.replace(/\//g, "");
	s = s.replace(/\(/g, "");
	s = s.replace(/\)/g, "");
	s = s.replace(/\=/g, "");

	return s;
}