# -*- coding: utf-8 -*-
# stock_analysis_vps.py
import os
import sys
import json
import argparse
import akshare as ak
import pandas as pd
import time
import numpy as np
from datetime import datetime, timedelta
import requests
import talib
from getpass import getpass
from openai import OpenAI
from openai import APIConnectionError
from deepseektest import get_market_analysis

# 获取当前文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    # 使用绝对路径
    INITIAL_SCREENING_PATH = os.path.join(BASE_DIR, "initial_screening.txt")
    RESULT_PATH = os.path.join(BASE_DIR, "amount_result.txt")
    TECHNICAL_PATH = os.path.join(BASE_DIR, "technical_result.txt")
    FINAL_RESULT_PATH = os.path.join(BASE_DIR, "final_selected.txt")

    # 从环境变量读取敏感信息
    PUSHPLUS_TOKEN = os.getenv("PUSHPLUS_TOKEN", "d1c91dc828e1430d92af54e58ca8c443")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-8a83960eb1df4bb08e48ba4e74a4a5be")
    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

    REQUEST_INTERVAL = 1.0
    MIN_DATA_LENGTH = 120
    WINDOW_RATIO = 0.2
    AMOUNT_RATIO = 1.8
    DEBUG_MODE = False  # 生产环境关闭调试模式


class StockAnalyzer:
    def __init__(self):
        self.today = datetime.now()
        self.start_date = self._calculate_start_date()
        self.quotation = ak.stock_zh_a_spot_em
        self.deepseek_client = OpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL,
            timeout=30.0
        )

    def getsc(self):
        result = get_market_analysis(
            stock_count=100,  # 只处理前50支热门股
            top_concepts=8  # 展示前5个概念
        )
        # 处理结果
        if result:

            print("\n=== 实时概念热度TOP8 ===")
            print(result['概念热度'].head(8))

            print("\n=== 概念龙头股TOP8 ===")
            print(result['龙头个股'].head(8))
        else:
            print("未获取到有效数据")
        return(result['概念热度'].head(8),result['龙头个股'].head(8))

    def deepseek_analysis(self, stocks_data):
        """深度分析（市场画像与个股分析独立）"""
        print("\n=== DeepSeek金融分析 ===")

        # 获取并校验市场数据
        market_analysis = self.getsc()
        concept_table, leader_table = pd.DataFrame(), pd.DataFrame()
        if market_analysis and len(market_analysis) == 2:
            concept_table, leader_table = market_analysis

        # 构建市场报告（带数据校验）
        market_report = []
        if not concept_table.empty and concept_table.shape[0] >= 8:  # 至少有3个有效概念
            market_report.append(f"### 实时概念热度TOP8\n{concept_table.head(8).to_markdown(index=False)}")
        if not leader_table.empty and '股票代码' in leader_table.columns:
            market_report.append(f"### 概念龙头股TOP8\n{leader_table.head(8).to_markdown(index=False)}")
        market_report = "## 市场画像（独立分析）\n" + "\n\n".join(market_report) if market_report else "## 市场画像\n暂无有效数据"
        self.send_market=market_report


        # 获取股票代码（带容错处理）
        stock_codes = [str(s.get('code', '')).split('.')[0] for s in stocks_data if s.get('code')]  # 统一格式处理
        stock_codes = list(set([c for c in stock_codes if c.isdigit() and len(c) == 6]))  # 去重并验证有效性

        # 获取实时数据（带股票代码过滤）
        def fetch_realtime_data(codes):
            """多线程获取实时数据"""
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def fetch_single(code):
                """带超时和重试的单个股票获取"""
                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        market = "sh" if code.startswith('6') else "sz"
                        # 优化点2：设置单个请求超时（8秒）
                        flow_data = ak.stock_individual_fund_flow(
                            stock=code, market=market
                        ).tail(5)
                        flow_data['股票代码'] = f"{code}.{market.upper()}"
                        return flow_data
                    except Exception as e:
                        if attempt == max_retries - 1:
                            print(f"股票{code}资金流获取失败: {str(e)}")
                            return None
                        time.sleep(1)  # 重试前等待

            # 多线程执行
            money_flow_dfs = []
            with ThreadPoolExecutor(max_workers=5) as executor:  # 控制并发数
                futures = {executor.submit(fetch_single, code): code for code in codes}
                for future in as_completed(futures, timeout=15):  # 总超时15秒
                    result = future.result()
                    if result is not None:
                        money_flow_dfs.append(result)

            # 合并数据（优化点3：限制数据量）
            money_flow = pd.concat(money_flow_dfs)[['股票代码', '日期', '主力净流入-净额', '超大单净流入-净额', '大单净流入-净额', '中单净流入-净额', '小单净流入-净额']] \
                if money_flow_dfs else pd.DataFrame()

            # 新闻数据获取（保持不变）
            return {"money_flow": money_flow}

        realtime_data = fetch_realtime_data(stock_codes)

        # 结构化用户消息内容
        flow_str = ""
        if not realtime_data['money_flow'].empty:
            # 按股票代码分组展示
            grouped_flow = realtime_data['money_flow'].groupby('股票代码')

            flow_str = "\n## 个股资金流向\n"
            for stock_code, group in grouped_flow:
                flow_str += f"### {stock_code}\n"
                flow_str += group.to_markdown(index=False) + "\n\n"

        user_content = f"""
        {market_report}

        === 个股分析数据 ===
        目标股票代码：{', '.join(stock_codes) or '无有效代码'}

        {flow_str}
        """

        # 重构后的系统提示词
        system_prompt = """作为A股中短线实战派专家，请按以下结构生成可操作性分析报告：

        # 市场画像分析（独立）
        1. 概念热度解读：分析资金聚集的持续性，识别伪热点
        2. 龙头股特征：行业分布/流通市值/技术形态共性
        3. 预警信号：过热概念或异常龙头股

        # 个股分析（独立）
        1. 资金验证：结合近5日数据，分析主力流向的持续性（重点观察：单日净流入超1亿/连续3日流入/占比突变超5%等关键信号）
        2. 热点关联度：
           - 显性关联：当前所属概念在市场热度TOP3中的匹配度
           - 潜在关联：业务可能延伸的热点领域（如：300134大富科技可关联5.5G/毫米波雷达等前沿概念）
        3. 立体化风险预警：
           - 估值维度：结合近三年PE/PB分位点（例：当前PE处于历史85%分位）
           - 事件驱动：未来1个月内的解禁明细（解禁量/成本价与现价差值）
           - 技术预警：重点观察20日均线支撑、MACD死叉、量价背离等信号
           - 筹码异动：股东户数变化率超±15%需特别警示
           - 短期策略：根据上述风险给出具体建议（例：跌破10.5元建议止损）

        <格式规范>
        1. 每个风险点必须包含量化指标和阈值判断
        2. 使用【关键信号】、【警戒线】等明确标记决策点
        3. 禁用模糊表述，如"关注"、"注意"等，改为具体建议"""

        try:
            print("🔍 正在生成深度报告...")

            # 智能重试配置
            max_retries = 3
            base_timeout = 60  # 基础超时30秒
            backoff_config = {
                'initial': 1.0,
                'factor': 1.8,
                'max_wait': 8.0
            }

            # 动态内容压缩（新增调试日志）
            compressed_content = user_content

            # 自适应超时重试循环
            analysis = None
            for attempt in range(max_retries + 1):
                current_timeout = base_timeout + attempt * 5  # 每次增加5秒
                try:
                    print(
                        f"🔄 尝试 {attempt + 1}/{max_retries + 1} | 超时：{current_timeout}s | 内容长度：{len(compressed_content)}字符")

                    # API请求（新增请求时间戳记录）
                    start_time = time.time()
                    response = self.deepseek_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": compressed_content}
                        ],
                        temperature=0.2,
                        max_tokens=8000,
                        stream=False,
                        timeout=current_timeout
                    )
                    elapsed_time = time.time() - start_time
                    print(f"✅ 请求成功 | 耗时：{elapsed_time:.1f}s")

                    # 处理响应（新增空值校验）
                    if response and response.choices:
                        analysis = response.choices[0].message.content

                        # 强制保存报告（新增校验）
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"DeepSeek_Analysis_{timestamp}.md"
                        try:
                            with open(filename, 'w', encoding='utf-8') as f:
                                f.write(f"# DeepSeek分析报告\n\n")
                                f.write(f"**生成时间：** {timestamp}\n\n")
                                f.write(f"## 输入参数\n{compressed_content[:1000]}...\n\n")
                                f.write(f"## 分析结果\n{analysis}")
                            print(f"📄 报告已保存至：{os.path.abspath(filename)}")
                        except Exception as save_error:
                            print(f"⚠️ 文件保存失败：{str(save_error)}")

                        # 控制台预览（新增关键指标提取）
                        print("\n=== 分析结果预览 ===")
                        preview_lines = [line for line in analysis.split('\n') if '★' in line or '风险' in line][:5]
                        print('\n'.join(preview_lines) or "无关键指标提取")
                        break  # 成功则退出重试循环

                    else:
                        print("⚠️ 收到空响应")
                        analysis = "未获取到有效分析结果"

                except APITimeoutError as e:
                    print(f"⏰ 请求超时：{str(e)}")
                    if attempt < max_retries:
                        wait_time = min(backoff_config['initial'] * (backoff_config['factor'] ** attempt),
                                        backoff_config['max_wait'])
                        print(f"⌛ 第{attempt + 1}次重试等待：{wait_time:.1f}s")
                        time.sleep(wait_time)
                        # 动态压缩内容（新增压缩比例提示）
                        new_length = int(len(compressed_content) * 0.7)
                        print(f"📉 内容长度从 {len(compressed_content)} 压缩至 {new_length}")
                        compressed_content = compressed_content[:new_length]
                    else:
                        print("⚠️ 达到最大重试次数，启用降级模式")
                        analysis = self.quick_analysis(stocks_data)

                except Exception as e:
                    print(f"‼️ 非超时异常：{str(e)}")
                    if Config.DEBUG_MODE:
                        import traceback
                        traceback.print_exc()
                    analysis = f"分析失败：{str(e)}"
                    break

            # 最终返回处理（新增空值保护）
            return analysis if analysis else "未生成有效分析报告"

        except Exception as outer_e:
            print(f"‼️ 外层异常：{str(outer_e)}")
            return f"系统错误：{str(outer_e)}"


    def _calculate_start_date(self):
        """计算历史数据起始日期（原AmountStrategy中的方法）"""
        required_days = int(Config.MIN_DATA_LENGTH / 4 * 1.3)
        return self.today - timedelta(days=required_days)

    def amount_analysis(self):
        """成交金额分析主入口（整合原main()函数逻辑）"""
        status_file = "process_status.json"
        current_date = self.today.strftime("%Y-%m-%d")

        # 读取股票列表
        with open(Config.INITIAL_SCREENING_PATH) as f:
            all_codes = [line.strip() for line in f if line.strip()]

        # 断点续传处理
        processed_codes = []
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status = json.load(f)
                if status['date'] == current_date:
                    processed_codes = status['processed']
                    print(f"检测到未完成任务，已处理 {len(processed_codes)} 支，剩余 {len(all_codes) - len(processed_codes)} 支")

        todo_codes = [c for c in all_codes if c not in processed_codes]
        results = []
        all_results = []
        total_passed = 0
        try:
            for idx, code in enumerate(todo_codes, 1):
                if self._analyze_single_stock(code):
                    results.append(code)
                    total_passed += 1
                    print(f"✅ 通过: {code}")
                else:
                    if Config.DEBUG_MODE:
                        print(f"❌ 未通过: {code}")
                # 更新进度
                processed_codes.append(code)

                # 定期保存进度（每50支或结束时）
                if idx % 50 == 0 or idx == len(todo_codes):
                    self._save_results(results, Config.RESULT_PATH)
                    all_results.extend(results)
                    results.clear()  # 清空当前批次
                    self._save_progress(status_file, current_date, processed_codes)
                    print(f"⏲️ 进度: 已处理 {len(processed_codes)} 支 ({idx / len(todo_codes):.1%})")

                time.sleep(Config.REQUEST_INTERVAL)

            # 最终清理
            if os.path.exists(status_file):
                os.remove(status_file)
            print(f"🎉 分析完成! 共筛选出  {total_passed} 支股票")
        except requests.exceptions.Timeout:
            print("‼️ API请求超时，请检查网络或减少分析范围")
            return "报告生成超时，建议减少股票数量后重试"
        except Exception as e:
            print(f"‼️ 分析异常：{str(e)}")
            return "报告生成失败，请联系技术支持"


    def _analyze_single_stock(self, code):
        """单个股票分析逻辑（原analyze_amount方法）"""
        try:
            # 获取金额数据
            amounts = self._get_amount_data(code)
            #amounts=code

            if len(amounts) < Config.MIN_DATA_LENGTH:
                if Config.DEBUG_MODE:
                    print(f"{code} 数据不足 ({len(amounts)}/{Config.MIN_DATA_LENGTH})")
                return False
            # 动态窗口计算
            window_size = 20


            # 计算指标
            historical_mean = amounts[:-window_size].mean()
            recent_mean = amounts[-window_size:].mean()

            # 调试输出
            if Config.DEBUG_MODE:
                print(
                    f"[{code}] 窗口:{window_size} 历史均额:{historical_mean / 1e4:.1f}万 "
                    f"近期均额:{recent_mean / 1e4:.1f}万 倍数:{recent_mean / historical_mean:.2f}x"
                )

            return recent_mean >= Config.AMOUNT_RATIO * historical_mean
        except Exception as e:
            if Config.DEBUG_MODE:
                print(f"分析 {code} 异常: {str(e)}")
            return False

    def _get_amount_data(self, code):
        """获取成交金额数据（仅使用东财接口，60分钟线）"""
        try:
            # 代码格式处理
            if '.' in code:
                pure_code = code.split('.')[0]
            else:
                pure_code = code.lstrip('shsz')  # 清理前缀

            # 计算时间范围（获取约160个60分钟周期）
            #end_date = datetime.now().strftime("%Y-%m-%d 15:00:00")  # 收盘时间
            #start_date = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d 09:30:00")  # 40

            # 调用东财接口（60分钟线）
            df = ak.stock_zh_a_hist_min_em(
                symbol=pure_code,
                period="60",  # 60分钟周期
                adjust="hfq",
                start_date="1979-09-01 09:32:00",  # 格式："YYYY-MM-DD"
                end_date="2222-01-01 09:32:00",
            )
            print(df)

            # 数据标准化
            df = self._process_data(df, 'em')

            print(df)


            # 验证数据量
            print(f"[DEBUG] {code} 获取 {len(df)} 条60分钟数据（预期≥120）")
            if len(df) < 100:
                print(f"⚠️ 数据不足: {code} 仅获取到 {len(df)} 条数据")
                return pd.Series()
            return df['amount'].iloc[-120:]  # 返回最近160个周期

        except Exception as e:
            print(f"❌ 严重错误: {code} 数据获取失败 - {str(e)}")
            return pd.Series()

    def _process_data(self, df, source):
        """处理东财分钟线数据格式（适配新接口返回的列名和时间格式）"""
        if df.empty:
            return df
        # 1. 列名重命名（根据实际返回的字段名）
        df = df.rename(columns={
            '时间': 'datetime',  # 时间列中文名
            '成交额': 'amount'  # 成交额列中文名
        })
        # 4. 按时间排序并设置索引
        return df.sort_values('datetime').set_index('datetime')




    def _save_results(self, results, path):
        """保存结果文件"""
        with open(path, 'a') as f:  # 改为追加模式
            f.write('\n'.join(results) + '\n')

    def _save_progress(self, path, date, processed):
        """保存处理进度"""
        with open(path, 'w') as f:
            json.dump({
                'date': date,
                'processed': processed,
                'timestamp': datetime.now().isoformat()
            }, f)


    def get_qiangrizhangfu(self, codes):
        """强势涨幅筛选（带详细日志）"""
        result = []
        total = len(codes)
        print(f"\n🔄 开始处理 {total} 支股票，预计需要 {total * 0.5 / 60:.1f} 分钟")
        start_time = time.time()

        for idx, code in enumerate(codes, 1):
            start_time = time.time()
            try:
                # 清洗股票代码格式
                pure_code = code.split('.')[0]
                market = 'sh' if code.endswith('XSHG') else 'sz'

                if Config.DEBUG_MODE:
                    print(f"\n[{idx}/{total}] 分析 {pure_code} [{market}]")

                # 获取后复权日线数据
                df = ak.stock_zh_a_hist(
                    symbol=pure_code,
                    period="daily",
                    adjust="hfq",
                    start_date="20240101",
                    end_date=self.today.strftime("%Y%m%d")
                )

                # 增强数据校验
                if df.empty:
                    print(f"❌ {code} 未获取到数据")
                    continue

                if len(df) < 30:
                    print(f"❌ {code} 数据不足（仅 {len(df)} 天）")
                    continue

                # 转换数据类型
                df = df.astype({
                    '开盘': 'float',
                    '收盘': 'float',
                    '成交量': 'float'
                })

                # 提取关键字段
                closes = df['收盘'].values
                opens = df['开盘'].values
                volumes = df['成交量'].values
                # 条件1: 最近三日阳线
                condition1 = closes[-1] > opens[-1] and closes[-2] > opens[-2] and closes[-3] > opens[-3]
                #condition1 = closes[-2] > opens[-2] and closes[-3] > opens[-3]
                if Config.DEBUG_MODE:
                    print(f"  → 前2日阳线: {'✅' if condition1 else '❌'}")

                # 条件2: 成交量递增
                condition2 = volumes[-1] > volumes[-2] or volumes[-2] > volumes[-3]
                if Config.DEBUG_MODE:
                    print(f"  → 量能增长: {'✅' if condition2 else '❌'}")

                # 条件3: 六日涨幅区间
                pct_change = (closes[-1] - closes[-6]) / closes[-6]
                condition3 = -0.05 <= pct_change <= 0.3
                if Config.DEBUG_MODE:
                    print(f"  → 六日涨幅: {pct_change:.2%} {'✅' if condition3 else '❌'}")

                #if sum([condition1, condition2, condition3]) >= 2:
                if condition1 and condition2 and condition3:

                    print(f"✅ [{code}] 通过筛选 | 用时 {time.time() - start_time:.1f}s")
                    result.append(code)
                else:
                    if Config.DEBUG_MODE:
                        print(f"  → 综合判定: ❌")

            except Exception as e:
                print(f"⚠️ 处理 {code} 异常: {str(e)}")

            time.sleep(0.5)
            if idx % 10 == 0:
                print(f"⏲️ 进度: {idx}/{total} ({idx / total:.0%}) 已找到 {len(result)} 支")

        print(f"\n🎯 筛选完成！通过 {len(result)} 支，淘汰率 {1 - len(result) / total:.0%}")
        return result

    def technical_analysis(self, codes):
        """技术指标分析（修复60参数错误）"""
        result = []
        qualified = []
        total = len(codes)
        print(f"\n🔄 开始技术分析 {total} 支股票，预计耗时: {total * 0.5 / 60:.1f} 分钟")

        for idx, code in enumerate(codes, 1):
            try:
                # 清洗股票代码（去除交易所后缀）
                pure_code = code.split('.')[0]
                print(f"\n[{idx}/{total}] 分析 {pure_code}")

                # 获取60分钟K线数据（正确使用分钟级接口）
                df = ak.stock_zh_a_hist_min_em(
                    symbol=pure_code,
                    period="60",
                    adjust='hfq',
                    start_date=(self.today - timedelta(days=60)).strftime("%Y-%m-%d"),
                    end_date=self.today.strftime("%Y-%m-%d")
                )

                # 数据校验
                if len(df) < 100:
                    print(f"❌ 数据不足 ({len(df)}/100)")
                    continue

                # 重命名列
                df = df.rename(columns={
                    '时间': 'datetime',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume'
                }).set_index('datetime')

                # 计算技术指标
                close_prices = df['close'].values
                high_prices = df['high'].values
                low_prices = df['low'].values

                # 威廉指标计算
                wr1_high = high_prices[-21:].max()
                wr1_low = low_prices[-21:].min()
                wr2_high = high_prices[-42:].max()
                wr2_low = low_prices[-42:].min()

                wr1 = 100 * (wr1_high - close_prices[-1]) / (wr1_high - wr1_low) if (wr1_high - wr1_low) != 0 else 100
                wr2 = 100 * (wr2_high - close_prices[-1]) / (wr2_high - wr2_low) if (wr2_high - wr2_low) != 0 else 100
                # MACD计算
                dif, dea, macd = talib.MACD(close_prices,
                                            fastperiod=12,
                                            slowperiod=26,
                                            signalperiod=9)

                conditions = [
                    wr1 < 60,  # WR1低于60
                    (wr1 - wr2) < 20,  # WR1与WR2差值小于20
                    dif[-1] > 0,  # DIF线上扬
                    dea[-1] > 0,  # DEA线上扬
                    macd[-1] > macd[-2] > macd[-3],  # MACD连续两日增长
                    macd[-3] < 0  or  macd[-4] < 0 or  macd[-5] < 0  or  macd[-6] < 0  # 前三天MACD为负值
                ]

                # 调试信息
                if Config.DEBUG_MODE:
                    debug_msg = f"""
                               [{code}] 技术指标:
                               WR%1: {wr1:.1f} | WR%2: {wr2:.1f} 
                               DIF: {dif[-1]:.2f} | DEA: {dea[-1]:.2f}
                               MACD序列: {macd[-3]:.2f} → {macd[-2]:.2f} → {macd[-1]:.2f}
                               条件验证: {" | ".join(str(c) for c in conditions)}
                               """
                    print(debug_msg)

                if all(conditions):
                    print(f"✅ {code} 通过技术筛选")
                    qualified.append(code)
                else:
                    print(f"❌ {code} 未通过技术条件")

            except Exception as e:
                print(f"⚠️ 技术分析异常 {code}: {str(e)}")
                if Config.DEBUG_MODE:
                    import traceback
                    traceback.print_exc()

            time.sleep(0.5)

        print(f"\n🎯 技术分析完成！通过 {len(result)} 支，淘汰率 {1 - len(result) / total:.0%}")
        return qualified

    def get_realtime_data(self, codes):
        """高效获取指定股票实时数据（直接请求）"""
        results = []

        for code in codes:
            pure_code = code.split('.')[0]  # 提前提取纯代码
            data_template = {
                "code": pure_code,
                "name": "N/A",
                "open": 0.0,
                "close": 0.0,
                "high": 0.0,
                "low": 0.0,
                "now": 0.0
            }

            try:
                # === 获取实时行情数据 ===
                df_bid_ask = ak.stock_bid_ask_em(symbol=pure_code)

                # === 获取股票名称 ===
                name = "N/A"
                try:
                    df_info = ak.stock_individual_info_em(symbol=pure_code)
                    name_row = df_info[df_info['item'] == '股票简称']
                    if not name_row.empty:
                        name = name_row['value'].values[0]
                except Exception as name_err:
                    if Config.DEBUG_MODE:
                        print(f"名称获取失败 {code}: {str(name_err)}")

                # === 构建数据字典 ===
                data = {
                    "code": pure_code,
                    "name": name,
                    "open": df_bid_ask[df_bid_ask['item'] == '今开']['value'].values[0],
                    "close": df_bid_ask[df_bid_ask['item'] == '昨收']['value'].values[0],
                    "high": df_bid_ask[df_bid_ask['item'] == '最高']['value'].values[0],
                    "low": df_bid_ask[df_bid_ask['item'] == '最低']['value'].values[0],
                    "now": df_bid_ask[df_bid_ask['item'] == '最新']['value'].values[0]
                }
                results.append(data)

            except Exception as e:
                if Config.DEBUG_MODE:
                    print(f"获取 {code} 数据失败: {str(e)}")
                results.append(data_template)

            # === 请求频率控制 ===
            time.sleep(0.3)

        return results

    def send_notification(self, data):
        """静默发送微信通知（令牌内置版）"""
        print("\n=== 开始推送通知 ===")

        if not data:
            print("⚠️ 无有效股票数据，跳过通知")
            return

        try:
            # 验证令牌配置
            if not hasattr(Config, 'PUSHPLUS_TOKEN') or not Config.PUSHPLUS_TOKEN:
                raise ValueError("未配置PUSHPLUS_TOKEN，请在Config类中设置")

            print("\n🔄 正在生成深度分析...")
            analysis_content = self.deepseek_analysis(data)

            # ========== 构建通知内容 ==========
            content = []

            # 1. 实时行情表格
            content.append("## 📈 实时行情")
            content.append("| 代码 | 名称 | 现价 | 涨跌幅 |\n|---|---|---|---|")
            for stock in data:
                chg_pct = (stock['now'] - stock['close']) / stock['close'] * 100
                content.append(f"| {stock['code']} | {stock['name']} | {stock['now']:.2f} | {chg_pct:.2f}% |")

            content.append(self.send_market)

            # 2. 深度分析报告
            content.append("\n## 🔍 深度分析")
            if analysis_content:
                content.append(analysis_content)
            else:
                content.append("⚠️ 深度分析获取失败，请查看日志")

            # 合并内容
            full_content = "\n".join(content)

            # ========== 发送请求 ==========
            response = requests.post(
                'http://www.pushplus.plus/send',
                json={
                    "token": Config.PUSHPLUS_TOKEN,
                    "title": f"{datetime.now():%Y-%m-%d} 智能选股报告",
                    "content": full_content,
                    "template": "markdown"
                },
                headers={'Content-Type': 'application/json'},
                timeout=15  # 延长超时时间
            )

            # 解析响应
            print(f"服务器响应状态码: {response.status_code}")
            if response.status_code == 200:
                resp_data = response.json()
                if resp_data.get('code') == 200:
                    print("✅ 推送成功！请查看微信")
                else:
                    print(f"❌ 推送失败: {resp_data.get('msg')}")
            else:
                print(f"‼️ 异常响应: {response.text}")

        except Exception as e:
            print(f"‼️ 发送失败: {str(e)}")
            if Config.DEBUG_MODE:
                import traceback
                traceback.print_exc()



class InteractiveConsole:
    def __init__(self):
        self.analyzer = StockAnalyzer()
        self.menu = {
            '1': {'name': '执行市值筛选', 'func': self.run_market_cap_screening},  # 新增的市值筛选
            '2': {'name': '执行成交金额筛选', 'func': self.run_amount_analysis},
            '3': {'name': '执行强势股筛选', 'func': self.run_qiangrizhangfu},
            '4': {'name': '执行技术指标分析', 'func': self.run_technical_analysis},
            '5': {'name': '获取实时行情数据', 'func': self.run_realtime_data},
            '6': {'name': '发送微信通知', 'func': self.run_send_notification},
            '7': {'name': '执行完整流程', 'func': self.run_full_process},
            '8': {'name': 'DeepSeek智能分析', 'func': self.run_deepseek_analysis},
            '0': {'name': '退出系统', 'func': exit}
        }


    def display_menu(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n=== 股票分析系统 ===")
        for k, v in self.menu.items():
            print(f"{k}. {v['name']}")
        print("===================")

    def run_market_cap_screening(self):
        """执行市值筛选（新增的步骤1）"""
        print("\n正在执行市值筛选（30-400亿，排除北交所/科创板）...")

        # 调用之前实现的筛选逻辑
        df = ak.stock_zh_a_spot_em()
        df["代码"] = df["代码"].astype(str).str.zfill(6)

        # 筛选条件
        condition = (
                (df["总市值"] >= 3e9) &
                (df["总市值"] <= 40e9) &
                (~df["代码"].str.startswith('8')) &
                (~df["代码"].str.startswith('688'))
        )
        filtered_df = df[condition]

        # 保存结果到初始文件（替换原有成交金额筛选的输入）
        filtered_df["代码"].to_csv(Config.INITIAL_SCREENING_PATH, index=False, header=False)
        print(f"筛选完成，共{len(filtered_df)}支，结果保存至{Config.INITIAL_SCREENING_PATH}")

    def get_selection(self):
        while True:
            choice = input("请选择操作编号: ").strip()
            if choice in self.menu:
                return choice
            print("无效输入，请重新选择")


    def run_deepseek_analysis(self):
        """触发DeepSeek分析流程"""
        if not os.path.exists(Config.FINAL_RESULT_PATH):
            print("请先完成筛选流程！")
            return

        # 获取实时数据
        with open(Config.FINAL_RESULT_PATH) as f:
            codes = [line.strip() for line in f]

        print("\n🔄 获取最新行情数据...")
        realtime_data = self.analyzer.get_realtime_data(codes)

        # 执行分析
        self.analyzer.deepseek_analysis(realtime_data)

    def run_amount_analysis(self):
        print("\n正在执行成交金额分析...")
        # 调用原有金额分析逻辑
        self.analyzer.amount_analysis()
        print(f"结果已保存到 {Config.RESULT_PATH}")

    def run_qiangrizhangfu(self):
        if not os.path.exists(Config.RESULT_PATH):
            print("请先执行成交金额筛选！")
            return

        with open(Config.RESULT_PATH) as f:
            codes = [line.strip() for line in f]

        print(f"\n正在对 {len(codes)} 支股票进行强势筛选...")
        result = self.analyzer.get_qiangrizhangfu(codes)

        with open(Config.TECHNICAL_PATH, 'w') as f:
            f.write('\n'.join(result))
        print(f"筛选完成，剩余 {len(result)} 支，结果保存到 {Config.TECHNICAL_PATH}")

    def run_technical_analysis(self):
        if not os.path.exists(Config.TECHNICAL_PATH):
            print("请先执行强势股筛选！")
            return

        with open(Config.TECHNICAL_PATH) as f:
            codes = [line.strip() for line in f]

        print(f"\n正在对 {len(codes)} 支股票进行技术分析...")
        result = self.analyzer.technical_analysis(codes)

        with open(Config.FINAL_RESULT_PATH, 'w') as f:
            f.write('\n'.join(result))
        print(f"分析完成，剩余 {len(result)} 支，结果保存到 {Config.FINAL_RESULT_PATH}")

    def run_realtime_data(self):
        if not os.path.exists(Config.FINAL_RESULT_PATH):
            print("请先完成技术分析！")
            return

        with open(Config.FINAL_RESULT_PATH) as f:
            codes = [line.strip() for line in f]

        print("\n正在获取实时数据...")
        data = self.analyzer.get_realtime_data(codes)

        print("\n最新行情数据：")
        print("{:<8} {:<10} {:<8} {:<8} {:<8} {:<8} {:<8}".format(
            "代码", "名称", "开盘价", "昨收", "最高", "最低", "现价"))
        for stock in data:
            print("{:<10} {:<10} {:<8.2f} {:<8.2f} {:<8.2f} {:<8.2f} {:<8.2f}".format(
                stock['code'], stock['name'], stock['open'],
                stock['close'], stock['high'], stock['low'], stock['now']))

    def run_send_notification(self):
        if not os.path.exists(Config.FINAL_RESULT_PATH):
            print("请先获取最终结果！")
            return

        with open(Config.FINAL_RESULT_PATH) as f:
            codes = [line.strip() for line in f]

        data = self.analyzer.get_realtime_data(codes)
        self.analyzer.send_notification(data)
        print("通知已发送！")

    def run_full_process(self):
        self.run_amount_analysis()
        self.run_qiangrizhangfu()
        self.run_technical_analysis()
        self.run_realtime_data()
        self.run_send_notification()

    def main_loop(self):
        while True:
            self.display_menu()
            choice = self.get_selection()
            self.menu[choice]['func']()
            input("\n按回车键继续...")


def parse_arguments():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(description='股票分析系统VPS版')

    parser.add_argument('--full-process',
                        action='store_true',
                        help='执行完整分析流程（量能+强势+技术分析）')

    parser.add_argument('--market-cap',
                        action='store_true',
                        help='执行市值筛选（30-400亿，排除北交所/科创板）')

    parser.add_argument('--amount',
                        action='store_true',
                        help='仅执行成交金额分析')

    parser.add_argument('--qiangrizhangfu',
                        action='store_true',
                        help='仅执行强势股筛选')

    parser.add_argument('--technical',
                        action='store_true',
                        help='仅执行技术指标分析')

    parser.add_argument('--send-report',
                        action='store_true',
                        help='仅发送通知报告')

    parser.add_argument('--debug',
                        action='store_true',
                        help='启用调试模式')

    return parser.parse_args()


def main():
    args = parse_arguments()
    analyzer = StockAnalyzer()

    # 设置调试模式
    if args.debug:
        Config.DEBUG_MODE = True
        print("=== 调试模式已启用 ===")

    try:
        # 完整流程
        if args.full_process:
            print("\n=== 开始完整分析流程 ===")
            analyzer.run_market_cap_screening()
            analyzer.amount_analysis()
            analyzer.run_qiangrizhangfu()
            analyzer.technical_analysis()
            analyzer.send_notification()
            return

        # 独立步骤
        if args.market_cap:
            analyzer.run_market_cap_screening()

        if args.amount:
            analyzer.amount_analysis()

        if args.qiangrizhangfu:
            analyzer.run_qiangrizhangfu()

        if args.technical:
            analyzer.technical_analysis()

        if args.send_report:
            analyzer.send_notification()

    except Exception as e:
        print(f"\n!!! 分析流程异常: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()