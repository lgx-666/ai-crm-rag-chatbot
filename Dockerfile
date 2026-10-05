# slim 版减体积；3.9 对齐你本地 Python 3.9.10，避免版本差异
FROM python:3.9-slim

# sentence-transformers/torch 运行时依赖 OpenMP 库 libgomp1，slim 镜像默认没有，
# 不装会在 import 时报 "libgomp.so.1: cannot open shared object file"
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 关键顺序：先只拷 requirements.txt 再装依赖。
# Docker 按层缓存——只要 requirements.txt 不变，改代码重新 build 时这几层走缓存，
# 不会每次都重装 torch 和几百个依赖。先拷代码再装依赖会让缓存每次失效。
COPY requirements.txt .
# 升级 pip，新版解析器更快更准
RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 本地 wheel 先装 torch：国内没有任何可用的新版 CPU 镜像源，只能手动下 wheel 再装。
# 三层设计，缺一就胖：
# 1) wheel 不放项目根，放 ../docker_wheels/，由 compose 的 additional_contexts 命名为 wheels。
# 2) 用 --mount=type=bind 而不是 COPY：COPY 会给这 184MB 单独留一层，装完即使 RUN rm 也
#    回收不掉（镜像层不可变）；bind mount 只在这个 RUN 期间可见，不进任何层。
# 3) .dockerignore 里排除 *.whl：实测被排除的文件连 --mount=type=bind 都取不到
#    （报 failed to compute cache key ... not found），所以必须靠命名上下文绕开，
#    否则 wheel 留在项目根就会被 COPY . . 第二次打进镜像。
# target 必须保留完整 wheel 文件名：pip 按 "名字-版本-py标签-abi-平台.whl" 校验，
# 挂成 /tmp/torch.whl 这种短名会直接报 Invalid wheel filename。
RUN --mount=type=bind,from=wheels,source=torch-2.8.0+cpu-cp39-cp39-manylinux_2_28_x86_64.whl,target=/tmp/torch-2.8.0+cpu-cp39-cp39-manylinux_2_28_x86_64.whl \
    pip install --no-cache-dir /tmp/torch-2.8.0+cpu-cp39-cp39-manylinux_2_28_x86_64.whl \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 再拷项目代码（.dockerignore 已挡掉 .env/data/tests/__pycache__）
COPY . .

# 声明端口（仅文档作用，真正的映射在 compose）
EXPOSE 5000

# 启动：python app.py 会先跑 init_data() 建表/填样例，再 app.run(host=0.0.0.0)
# 注：生产更推荐 gunicorn，但你的 /chat 是 SSE 流式，gunicorn 要配 --worker-class gevent
# 才能正确逐 token 推流；作品集演示阶段先用 python app.py 跑通
CMD ["python", "app.py"]