import asyncio
import random

# 模拟计算函数
async def compute():
    print("[计算任务] 计算开始...")
    await asyncio.sleep(60)  # 模拟60秒计算
    if random.random() < 0.8:
        return {"status": "success", "message": "计算完成"}
    else:
        return {"status": "failed", "message": "计算失败"}

# 模拟结果提取函数
async def extract_result():
    await asyncio.sleep(1)  # 模拟API调用延迟
    if random.random() < 0.7:
        return {"status": "success", "message": "提取成功"}
    return {"status": "failed", "message": "提取失败"}

async def main():
    # 创建停止事件和共享状态
    stop_event = asyncio.Event()
    compute_task = asyncio.create_task(compute())

    # 定期提取任务
    async def periodic_extract():
        while not stop_event.is_set():
            print("[提取任务] 正在提取中...")
            response = await extract_result()
            
            if response["status"] == "success":
                print(f"[提取结果] {response['message']}")
                stop_event.set()
                break
            
            print(f"[提取结果] {response['message']}")
            await asyncio.sleep(5)

    # 启动并行任务
    extract_task = asyncio.create_task(periodic_extract())

    # 处理计算结果
    try:
        compute_result = await compute_task
        print(f"[计算结果] {compute_result['message']}")
        
        if compute_result["status"] == "failed":
            stop_event.set()
    finally:
        # 等待提取任务正常结束
        if not extract_task.done():
            await extract_task

if __name__ == "__main__":
    asyncio.run(main())