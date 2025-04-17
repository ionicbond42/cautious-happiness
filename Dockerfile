# 使用官方Python镜像作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（包括字体）
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    fonts-wqy-zenhei \
    fonts-wqy-microhei \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements.txt并安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 设置环境变量，标识当前在Docker环境中运行
ENV DOCKER_ENV=true

# 设置字体路径环境变量
ENV FONT_PATH=/usr/share/fonts/truetype/msttcorefonts/msyh.ttc

# 创建必要的目录
RUN mkdir -p /app/models/saved_models \
    && mkdir -p /app/reports/feature_analysis \
    && mkdir -p /app/data/generated

# 设置容器启动命令
CMD ["python", "models/risk_model.py"]