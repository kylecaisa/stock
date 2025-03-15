import akshare as ak
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import time
import random


# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)


class HotStockAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger('HotStockAnalyzer')
        self.logger.setLevel(logging.DEBUG)

    def fetch_hot_stocks(self):
        """获取实时热门股票（增强校验版）"""
        try:
            self.logger.info("正在获取东方财富实时热度榜...")

            # 获取原始数据
            df = ak.stock_hot_rank_em()
            self.logger.debug(f"原始数据字段: {df.columns.tolist()}")

            # 校验必要字段存在
            if '代码' not in df.columns:
                raise KeyError("数据中缺少'代码'字段")

            # 提取前100股票代码
            raw_codes = df['代码'].astype(str).str.strip().tolist()[:100]
            self.logger.info(f"成功获取{len(raw_codes)}支股票")
            return raw_codes

        except Exception as e:
            self.logger.error(f"获取热门股票失败: {str(e)}", exc_info=True)
            return []

    def fetch_keywords(self, stock_codes):
        """获取股票关键词（带重试机制）"""
        results = []
        failed_codes = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(self._get_single_keyword, code): code for code in stock_codes}

            for future in as_completed(futures):
                code = futures[future]
                try:
                    data = future.result()
                    if data:
                        results.extend(data)
                        self.logger.info(f"{code} 采集成功，获取{len(data)}个关键词")
                    else:
                        failed_codes.append(code)
                except Exception as e:
                    self.logger.warning(f"{code} 采集失败: {str(e)}")
                    failed_codes.append(code)
                # 添加随机延迟防止封禁
                time.sleep(random.uniform(0.5, 1.2))

        self.logger.info(f"关键词采集完成 | 成功: {len(results)} | 失败: {len(failed_codes)}")
        return pd.DataFrame(results, columns=['股票代码', '概念名称', '热度']) if results else pd.DataFrame()

    def _get_single_keyword(self, code):
        """获取单个股票的关键词（带重试）"""
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                df = ak.stock_hot_keyword_em(symbol=code)
                self.logger.debug(f"{code} 原始返回字段: {df.columns.tolist() if not df.empty else '空数据'}")

                # 校验数据格式
                if df.empty:
                    return []
                if not {'概念名称', '热度'}.issubset(df.columns):
                    raise KeyError(f"缺失必要字段，现有字段: {df.columns.tolist()}")

                # 清洗数据
                return [(code, row['概念名称'], row['热度']) for _, row in df.iterrows()]

            except Exception as e:
                if attempt == max_retries:
                    raise
                self.logger.warning(f"{code} 第{attempt + 1}次重试...")
                time.sleep(1.5 ** attempt)
                return []

class MarketAnalyzer:
     def analyze(self, keyword_df):
         """执行市场分析"""
         if keyword_df.empty:
             return {}

         return {
             "概念热度": self._concept_heat_analysis(keyword_df),
             "龙头个股": self._stock_leader_analysis(keyword_df)
         }

     def _concept_heat_analysis(self, df):
         """生成概念热度报告"""
         concept_stats = df.groupby('概念名称').agg(
             总热度=('热度', 'sum'),
             涉及股票数=('股票代码', 'nunique'),
             股票列表=('股票代码', lambda x: list(x.unique()))
         ).reset_index().sort_values('总热度', ascending=False)

         return concept_stats.head(10)

     def _stock_leader_analysis(self, df):
         """识别概念龙头股"""
         leaders = df.loc[df.groupby('概念名称')['热度'].idxmax()]
         return leaders[['概念名称', '股票代码', '热度']].rename(
             columns={'热度': '最高热度'}
         ).sort_values('最高热度', ascending=False)

     # 使用示例


def get_market_analysis(stock_count=100, top_concepts=10):
    """对外暴露的主接口函数（修正版）"""
    stock_fetcher = HotStockAnalyzer()
    analyzer = MarketAnalyzer()

    hot_codes = stock_fetcher.fetch_hot_stocks()
    if not hot_codes:
        return {}

    # 获取完整关键词数据
    keyword_data = stock_fetcher.fetch_keywords(hot_codes[:stock_count])

    # 修正：直接使用完整数据
    return analyzer.analyze(keyword_data)  # 移除head()


