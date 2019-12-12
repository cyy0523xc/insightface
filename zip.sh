#!/bin/bash
# 
# 
# Author: alex
# Created Time: 2019年12月12日 星期四 16时20分17秒
#!/bin/bash
# 
# 将项目打包压缩
# Author: alex
# Created Time: 2019年03月11日 星期一 10时38分16秒
cd ../
if [ ! -d insightface ]; then
    echo "$PWD: 当前目录错误."
fi
version=
if [ $# = 1 ]; then 
    version="-$1"
fi

date_str=`date -I`
filename=insightface-"${date_str//-/}$version".zip
if [ -f "$filename" ]; then
    rm -f "$filename"
fi

    #-x "*/server/*" \
zip -r "$filename" insightface \
    -x "*/.git/*" \
    -x "*/.*" \
    -x "*/*/*.swp" \
    -x "*/__pycache__/*" 

# 发到测试机
scp "$filename" ibbd@192.168.80.245:~/gf/
