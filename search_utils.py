# search_utils.py
import os
import requests
import json
from config import BOCHA_API_KEY
from utils.logging_config import setup_logging

logger = setup_logging()

def bocha_search(query: str, count: int = 5) -> dict:
    """
    调用博查搜索API，返回搜索结果
    """
    if not BOCHA_API_KEY:
        logger.error("博查API Key未配置")
        return {"error": "API Key未配置"}
    
    url = "https://api.bocha.cn/v1/web-search"
    headers = {
        'Authorization': f'Bearer {BOCHA_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        "query": query,
        "summary": True,
        "count": count
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        logger.error("博查搜索超时")
        return {"error": "请求超时"}
    except requests.exceptions.RequestException as e:
        logger.error(f"博查搜索请求失败: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"博查搜索异常: {e}")
        return {"error": str(e)}