from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os
from calculate import generate_sample_curves
from calculate import calculate_cosine_similarity
from pydantic import BaseModel
from typing import List

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 创建FastAPI应用实例
app = FastAPI(
    title="曲线数据API",
    description="生成曲线数据和计算余弦相似度的API服务",
    version="1.0.0"
)

# 添加CORS中间件，允许前端页面访问API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)


# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=current_dir), name="static")

# 设置根路由返回HTML页面
@app.get("/")
async def read_root():
    return FileResponse(os.path.join(current_dir, "curve_visualization.html"))

# 定义请求数据模型
class CurveData(BaseModel):
    curve1: List[float]  # 第一条曲线的数据点列表
    curve2: List[float]  # 第二条曲线的数据点列表

# 生成曲线数据的端点
@app.get("/generate-curves")
async def generate_curves():
    try:
        # 调用request_demo1.py中的函数生成曲线数据
        curve1, curve2 = generate_sample_curves()
        return {"curve1": curve1, "curve2": curve2}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成曲线数据时发生错误: {str(e)}")

# 计算余弦相似度的端点
@app.post("/calculate/cosine-similarity")
async def calculate_similarity(data: CurveData):
    try:
        # 调用calculate.py中的函数计算余弦相似度
        similarity = calculate_cosine_similarity(data.curve1, data.curve2)
        return {"cosine_similarity": similarity}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"计算过程中发生错误: {str(e)}")

# 启动服务器
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)