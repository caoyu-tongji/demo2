# FastAPI的demo示例
from fastapi import FastAPI, HTTPException, Query, Path, Body
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

# 创建FastAPI应用实例
app = FastAPI(
    title="FastAPI演示",
    description="这是一个FastAPI框架的基本演示应用",
    version="0.1.0"
)

# 定义数据模型
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

# 内存中的项目存储
items = {}

# 根路径
@app.get("/")
async def read_root():
    return {"message": "欢迎使用FastAPI！"}

# 获取所有项目
@app.get("/items/", response_model=List[Item])
async def read_items():
    return list(items.values())

# 获取单个项目
@app.get("/items/{item_id}", response_model=Item)
async def read_item(item_id: int = Path(..., title="要获取的项目ID", ge=0)):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="项目未找到")
    return items[item_id]

# 创建项目
@app.post("/items/", response_model=Item, status_code=201)
async def create_item(item: Item):
    item_id = len(items)
    items[item_id] = item
    return item

# 更新项目
@app.put("/items/{item_id}", response_model=Item)
async def update_item(item_id: int, item: Item):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="项目未找到")
    items[item_id] = item
    return item

# 删除项目
@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="项目未找到")
    del items[item_id]
    return {"message": "项目已删除"}

# 查询参数示例
@app.get("/search/")
async def search_items(q: Optional[str] = Query(None, min_length=3, max_length=50)):
    if q:
        return [item for item in items.values() if q.lower() in item.name.lower()]
    return list(items.values())

# 运行服务器（当直接运行此脚本时）
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)