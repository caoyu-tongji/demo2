from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import uvicorn
from calculate import calculate_cosine_similarity
import os

# 创建FastAPI应用实例，设置API文档信息
app = FastAPI(
    title="曲线余弦相似度计算API",
    description="计算两条曲线的余弦相似度的Web API服务",
    version="1.0.0"
)

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 挂载静态文件目录
app.mount("/", StaticFiles(directory=current_dir, html=True), name="static")

# 定义请求数据模型
class CurveData(BaseModel):
    curve1: List[float]  # 第一条曲线的数据点列表
    curve2: List[float]  # 第二条曲线的数据点列表

# POST请求处理端点
# 客户端发送请求时会记录开始时间，服务器处理完成后记录结束时间
# 总响应时间 = 结束时间 - 开始时间，包含了：
# 1. 网络传输时间（请求和响应）
# 2. 服务器处理时间（数据验证和计算）
@app.post("/calculate/cosine-similarity")
async def calculate_similarity(data: CurveData):
    try:
        # 调用计算函数处理数据
        similarity = calculate_cosine_similarity(data.curve1, data.curve2)
        # 返回计算结果
        return {"cosine_similarity": similarity}
    except ValueError as e:
        # 处理输入数据验证错误（如数据长度不匹配）
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 处理其他未预期的错误
        raise HTTPException(status_code=500, detail=f"计算过程中发生错误: {str(e)}")

# 启动服务器
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)