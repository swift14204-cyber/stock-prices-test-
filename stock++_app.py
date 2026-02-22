import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier # 切换为分类器
from sklearn.metrics import accuracy_score

# --- 1. 配置与映射 ---
st.set_page_config(page_title="AI 量化信号实验室", layout="wide")
st.title("🏹 涨跌概率预测模型 (分类信号版)")

KEYWORD_MAP = {"SMIC": "0981.HK", "中芯国际": "0981.HK", "拉美基金": "0P00000PTG.F"}

def get_symbol(query):
    q = query.strip().upper()
    if q in KEYWORD_MAP: return KEYWORD_MAP[q]
    try:
        search = yf.Search(query, max_results=1)
        return search.quotes[0]['symbol'] if (search and search.quotes) else q
    except: return q

# --- 2. 增强特征工程 (加入 Lags 滞后特征) ---
def add_features(data):
    d = data.copy()
    if 'Volume' in d.columns: d['Volume'] = d['Volume'].fillna(0)
    
    # 基础指标
    d['RSI'] = ta.rsi(d['Close'], length=14)
    d['SMA_20'] = ta.sma(d['Close'], length=20)
    d['SMA_50'] = ta.sma(d['Close'], length=50)
    d['Return'] = d['Close'].pct_change()
    
    # 加入滞后特征：让 AI 看到过去 3 天发生了什么
    for i in range(1, 4):
        d[f'Lag_Return_{i}'] = d['Return'].shift(i)
    
    d['DayOfWeek'] = d.index.dayofweek
    return d

user_input = st.text_input("请输入股票/基金代码", value="AAPL")

if user_input:
    symbol = get_symbol(user_input)
    raw_df = yf.Ticker(symbol).history(period="5y")

    if not raw_df.empty and len(raw_df) > 150:
        df = add_features(raw_df)
        
        # --- 3. 定义分类目标：明天是否上涨 ---
        # 1 = 上涨 (> 0), 0 = 下跌或平盘
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df_ml = df.dropna().copy()

        # 特征选择
        features = ['RSI', 'SMA_20', 'SMA_50', 'Return', 'Lag_Return_1', 'Lag_Return_2', 'Lag_Return_3', 'DayOfWeek']
        X = df_ml[features]
        y = df_ml['Target']

        # 训练集划分
        split = len(df_ml) - 150
        model = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=42)
        model.fit(X[:split], y[:split])

        # 回测准确率
        y_pred = model.predict(X[split:])
        acc = accuracy_score(y[split:], y_pred)

        # --- 4. 未来 14 天概率预测 ---
        future_dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=14)
        future_probs = [] # 存储上涨概率
        
        temp_df = df.copy()
        for f_date in future_dates:
            current_df = add_features(temp_df)
            last_row = current_df[features].iloc[-1:].copy()
            
            # 获取上涨的概率 [P(0), P(1)]
            prob_up = model.predict_proba(last_row)[0][1]
            future_probs.append(prob_up)
            
            # 模拟新的一天价格 (基于概率假设)
            sim_ret = 0.005 if prob_up > 0.5 else -0.005
            new_price = temp_df['Close'].iloc[-1] * (1 + sim_ret)
            new_row = pd.DataFrame({'Close': new_price, 'Volume': 0}, index=[f_date])
            temp_df = pd.concat([temp_df, new_row])

        # --- 5. 绘图与可视化 ---
        fig = go.Figure()

        # 历史价格
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="历史价格/净值", line=dict(color='#00FF00', width=2)))
        
        # 均线
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name="20日支撑线", line=dict(color='orange', width=1, dash='dot')))

        # 未来趋势 (基于概率模拟)
        future_series = pd.Series(temp_df['Close'].iloc[-14:], index=future_dates)
        fig.add_trace(go.Scatter(x=future_series.index, y=future_series, name="预期趋势演进", line=dict(color='#FF4500', width=3)))

        fig.update_layout(
            height=650, template="plotly_dark",
            title=f"{symbol} 趋势信号分析 (模型回测准确率: {acc:.1%})",
            xaxis=dict(rangeslider=dict(visible=True)),
            hovermode="x unified"
        )
        fig.update_xaxes(range=[df.index[-120], future_dates[-1]])
        st.plotly_chart(fig, use_container_width=True)

        # --- 6. 核心决策面板 ---
        st.markdown("### 🚦 AI 实时交易信号")
        c1, c2, c3 = st.columns(3)
        
        current_prob = future_probs[0]
        if current_prob > 0.6:
            signal = "🟢 强烈看涨"
            advice = "模型识别到明显的上升波段特征，上涨概率极高。"
        elif current_prob > 0.52:
            signal = "🟡 偏向看多"
            advice = "有一定上攻动力，但需注意量能配合。"
        elif current_prob < 0.4:
            signal = "🔴 强烈看跌"
            advice = "技术形态走坏，模型提示下行风险。"
        else:
            signal = "⚪ 观望震荡"
            advice = "多空平衡，建议等待信号明确。"

        c1.metric("明日上涨概率", f"{current_prob:.1%}")
        c2.subheader(f"综合建议: {signal}")
        c3.info(advice)
        
        st.caption(f"注：回测准确率 {acc:.1%} 仅代表过去 150 天模型对该标的的识别能力。")
    else:
        st.error("数据不足，无法运行分类模型。")
