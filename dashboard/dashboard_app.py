
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

DB_PATH = r'/content/drive/MyDrive/StudentRiskProject/student_risk.db'
MODEL_PATH = r'/content/drive/MyDrive/StudentRiskProject/best_model.pkl'
FEAT = ["year","cgpa","previous_failures","assignment_completion",
        "attendance_rate","unexcused_absences","behavioral_incidents",
        "days_since_incident","scholarship","extracurricular",
        "hostel_resident","part_time_job"]

st.set_page_config(page_title="Student Risk Prediction", page_icon="🎓", layout="wide")

st.markdown('''
<style>
    .stApp { background-color: #0A0E27; }
    .main-header {
        background: linear-gradient(135deg, #1a1f3a 0%, #2d3561 100%);
        padding: 2rem; border-radius: 12px; color: #ffffff; text-align: center;
        margin-bottom: 2rem; border: 1px solid #2563eb;
    }
    .main-header h1 { font-size: 2.5rem; font-weight: bold; margin: 0; color: #ffffff; }
    .main-header p { font-size: 1.1rem; margin: 0.5rem 0 0 0; color: #93c5fd; }
    .metric-card {
        background: #1a1f3a; padding: 1.5rem; border-radius: 12px;
        border: 1px solid #2d3561; border-left: 4px solid #3b82f6; transition: all 0.3s ease;
    }
    .metric-card:hover { transform: translateY(-3px); border-color: #3b82f6; box-shadow: 0 8px 24px rgba(59, 130, 246, 0.2); }
    .metric-card h3 { color: #93c5fd; font-size: 0.75rem; font-weight: 600; margin: 0 0 0.5rem 0; text-transform: uppercase; letter-spacing: 1.5px; }
    .metric-card .value { color: #ffffff; font-size: 2rem; font-weight: bold; margin: 0; }
    .metric-card .delta { color: #6b7280; font-size: 0.85rem; margin-top: 0.3rem; }
    .student-card {
        background: #1a1f3a; padding: 1.2rem; border-radius: 10px; margin: 0.7rem 0;
        border: 1px solid #2d3561; border-left: 4px solid #6b7280; transition: all 0.3s ease; cursor: pointer;
    }
    .student-card:hover { border-color: #3b82f6; box-shadow: 0 4px 16px rgba(59, 130, 246, 0.15); transform: translateX(5px); }
    .student-card.high-risk { border-left-color: #ef4444; background: linear-gradient(to right, rgba(239, 68, 68, 0.05), #1a1f3a); }
    .student-card.medium-risk { border-left-color: #f59e0b; background: linear-gradient(to right, rgba(245, 158, 11, 0.05), #1a1f3a); }
    .student-card.low-risk { border-left-color: #10b981; background: linear-gradient(to right, rgba(16, 185, 129, 0.05), #1a1f3a); }
    .student-card h3 { color: #ffffff; margin: 0; font-size: 1.1rem; }
    .student-card p { color: #9ca3af; margin: 0.3rem 0; font-size: 0.9rem; }
    .risk-badge {
        display: inline-block; padding: 0.4rem 1rem; border-radius: 20px; font-weight: bold;
        font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px;
    }
    .risk-high { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid #ef4444; }
    .risk-medium { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
    .risk-low { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid #10b981; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; background-color: #1a1f3a; border: 1px solid #2d3561;
        border-radius: 8px 8px 0 0; padding: 10px 20px; font-weight: 600; color: #9ca3af;
    }
    .stTabs [data-baseweb="tab"]:hover { background-color: #252b4a; color: #ffffff; }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: #ffffff; border-color: #3b82f6;
    }
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); color: white;
        border: none; border-radius: 8px; padding: 0.6rem 1.5rem; font-weight: 600;
    }
    .stButton > button:hover { box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4); transform: translateY(-2px); }
    section[data-testid="stSidebar"] { background-color: #0f1729; border-right: 1px solid #2d3561; }
    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }
    p, span, label { color: #d1d5db !important; }
    hr { border-color: #2d3561 !important; }
    .section-card { background: #1a1f3a; border-radius: 14px; padding: 1.4rem 1.6rem; border: 1px solid #2d3561; margin-bottom: 1.2rem; }
    .section-title { font-size: 1rem; font-weight: 700; color: #93c5fd; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #2d3561; }
    .result-card-high { background: linear-gradient(135deg, #1a0505, #3b0a0a); border: 3px solid #ef4444; border-radius: 20px; padding: 3rem; text-align: center; max-width: 480px; width: 90%; box-shadow: 0 0 60px rgba(239,68,68,0.4); animation: popIn 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); }
    .result-card-medium { background: linear-gradient(135deg, #1a1205, #3b2a0a); border: 3px solid #f59e0b; border-radius: 20px; padding: 3rem; text-align: center; max-width: 480px; width: 90%; box-shadow: 0 0 60px rgba(245,158,11,0.4); animation: popIn 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); }
    .result-card-low { background: linear-gradient(135deg, #051a0a, #0a3b1a); border: 3px solid #10b981; border-radius: 20px; padding: 3rem; text-align: center; max-width: 480px; width: 90%; box-shadow: 0 0 60px rgba(16,185,129,0.4); animation: popIn 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); }
    @keyframes popIn { 0% { transform: scale(0.5); opacity: 0; } 100% { transform: scale(1); opacity: 1; } }
    .result-score { font-size: 4rem; font-weight: 900; margin: 0.5rem 0; }
    .result-label { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem; }
    .result-desc  { font-size: 0.95rem; color: #d1d5db; margin-top: 0.8rem; line-height: 1.6; }
    .factor-pill { display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600; margin: 0.3rem; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #ffffff; }
</style>
''', unsafe_allow_html=True)

st.markdown('''
<div class="main-header">
    <h1>🎓 Student Risk Prediction System</h1>
    <p>AI-Powered Early Warning System for Academic Intervention</p>
</div>
''', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data(ttl=60)
def load_data():
    q = '''SELECT s.student_id,s.student_name,s.year,s.school,s.program,
        CAST(ar.cgpa AS REAL) as cgpa,
        CAST(ar.previous_failures AS INTEGER) as previous_failures,
        CAST(ar.assignment_completion AS REAL) as assignment_completion,
        CAST(COALESCE((SELECT ROUND(SUM(CASE WHEN a.status='Present' THEN 1.0 ELSE 0 END)/30,3)
                  FROM attendance a WHERE a.student_id=s.student_id),0.85) AS REAL) as attendance_rate,
        CAST(COALESCE((SELECT SUM(a.unexcused) FROM attendance a WHERE a.student_id=s.student_id),0) AS INTEGER) as unexcused_absences,
        CAST(COALESCE(bi.total_incidents,0) AS INTEGER) as behavioral_incidents,
        CAST(COALESCE(bi.days_since_last,365) AS INTEGER) as days_since_incident,
        CAST(COALESCE(sd.scholarship,0) AS INTEGER) as scholarship,
        CAST(COALESCE(sd.extracurricular,0) AS INTEGER) as extracurricular,
        CAST(COALESCE(sd.hostel_resident,0) AS INTEGER) as hostel_resident,
        CAST(COALESCE(sd.part_time_job,0) AS INTEGER) as part_time_job
    FROM students s
    LEFT JOIN academic_records ar ON s.student_id=ar.student_id
    LEFT JOIN behavioral_incidents bi ON s.student_id=bi.student_id
    LEFT JOIN student_demographics sd ON s.student_id=sd.student_id
    LEFT JOIN counselor_referrals cr ON s.student_id=cr.student_id
    WHERE ar.cgpa IS NOT NULL'''
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    df = pd.read_sql_query(q, conn)
    conn.close()
    for col in FEAT:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    df = df[df['cgpa'].notna() & (df['cgpa'] >= 0) & (df['cgpa'] <= 10.0)]
    pkg = load_model()
    X = df[FEAT].copy()
    for col in FEAT: X[col] = X[col].astype(float)
    try:
        X_scaled = pkg['scaler'].transform(X)
        probs = pkg['model'].predict_proba(X_scaled)[:, 1]
    except:
        probs = np.zeros(len(df))
    df['risk_probability'] = probs
    df['risk_level'] = pd.cut(probs, bins=[0, .35, .65, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk'], include_lowest=True).astype(str)
    return df

with st.sidebar:
    st.markdown("### 📊 Model Information")
    pkg = load_model()
    model_type    = pkg.get('model_type', 'Unknown')
    training_size = pkg.get('training_samples', 200)
    perf          = pkg.get('performance', {})
    accuracy      = perf.get('Accuracy', 0.0)
    st.markdown(f'<div class="metric-card"><h3>Model Type</h3><div class="value" style="font-size:1.5rem;">{model_type}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-card" style="margin-top:0.8rem;"><h3>Accuracy</h3><div class="value" style="font-size:1.3rem;">{accuracy:.1%}</div><div class="delta">Trained on {training_size} students</div></div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("### 🎯 Filters")
    school_filter = st.multiselect("School", ['VSST','VSOD','VSOL','JAGSOM','TSM'], default=['VSST','VSOD','VSOL','JAGSOM','TSM'])
    year_filter   = st.multiselect("Year", [1,2,3,4], default=[1,2,3,4], format_func=lambda x: f"Year {x}")
    risk_filter   = st.multiselect("Risk Level", ['Low Risk','Medium Risk','High Risk'], default=['Low Risk','Medium Risk','High Risk'])
    st.divider()
    with st.expander("📚 School Information"):
        st.markdown('''
**VSST** - Science & Technology: B.Tech CSE, B.Tech AI/ML
**VSOD** - Design: B.Des Fashion, Product, Communication
**VSOL** - Law: BA LLB, BBA LLB, LLB
**JAGSOM** - Business: BBA, MBA
**TSM** - Music: B.Mus Performance, Production, Composition, B.Tech Sound Engineering
''')
    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

df = load_data()
df_filtered = df[
    (df['school'].isin(school_filter)) &
    (df['year'].isin(year_filter)) &
    (df['risk_level'].isin(risk_filter))
]

t1,t2,t3,t4,t5,t6,t7 = st.tabs(["📊 Overview","👥 Students","📈 Analytics","🧠 Model Insights","⚠️ Alerts","🔮 Predict","➕ Add Student"])

with t1:
    total=len(df); high=len(df[df['risk_level']=='High Risk']); medium=len(df[df['risk_level']=='Medium Risk']); avg_cgpa=df['cgpa'].mean()
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><h3>Total Students</h3><div class="value">{total}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card" style="border-left-color:#ef4444;"><h3>High Risk</h3><div class="value" style="color:#fca5a5;">{high}</div><div class="delta">{high/total*100:.1f}%</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card" style="border-left-color:#f59e0b;"><h3>Medium Risk</h3><div class="value" style="color:#fbbf24;">{medium}</div><div class="delta">{medium/total*100:.1f}%</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card" style="border-left-color:#10b981;"><h3>Avg CGPA</h3><div class="value" style="color:#6ee7b7;">{avg_cgpa:.2f}</div></div>', unsafe_allow_html=True)
    st.divider()
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("📊 Risk Distribution")
        rc = df['risk_level'].value_counts()
        fig = go.Figure(data=[go.Pie(labels=rc.index, values=rc.values, hole=0.5,
            marker=dict(colors=['#10b981','#f59e0b','#ef4444']), textinfo='label+percent', textfont=dict(size=13,color='white'))])
        fig.update_layout(showlegend=True, height=350, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#ffffff'))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("📈 Risk by Year")
        yr = df.groupby(['year','risk_level']).size().unstack(fill_value=0)
        fig = go.Figure()
        for risk,color in [('Low Risk','#10b981'),('Medium Risk','#f59e0b'),('High Risk','#ef4444')]:
            if risk in yr.columns:
                fig.add_trace(go.Bar(name=risk, x=yr.index, y=yr[risk], marker_color=color, text=yr[risk], textposition='auto'))
        fig.update_layout(barmode='stack', height=350, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#ffffff'))
        st.plotly_chart(fig, use_container_width=True)

with t2:
    st.subheader(f"👥 Student List ({len(df_filtered)} students)")
    search = st.text_input("🔍 Search","")
    if search: df_filtered = df_filtered[df_filtered['student_name'].str.contains(search,case=False,na=False)]
    for _,row in df_filtered.head(20).iterrows():
        rc = 'high-risk' if row['risk_level']=='High Risk' else 'medium-risk' if row['risk_level']=='Medium Risk' else 'low-risk'
        bc = 'risk-high' if row['risk_level']=='High Risk' else 'risk-medium' if row['risk_level']=='Medium Risk' else 'risk-low'
        st.markdown(f'''<div class="student-card {rc}"><div style="display:flex;justify-content:space-between;">
            <div><h3>{row['student_name']}</h3><p>Year {int(row['year'])} | CGPA: {row['cgpa']:.2f}</p></div>
            <div><span class="risk-badge {bc}">{row['risk_level']}</span><p style="color:#3b82f6;font-weight:bold;">{row['risk_probability']:.0%}</p></div>
        </div></div>''', unsafe_allow_html=True)

with t3:
    st.subheader("📈 Predictive Analytics")
    col1,col2 = st.columns(2)
    with col1:
        fig = px.scatter(df, x='cgpa', y='risk_probability', color='risk_level',
            color_discrete_map={'Low Risk':'#10b981','Medium Risk':'#f59e0b','High Risk':'#ef4444'}, height=350)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,31,58,0.3)', font=dict(color='#ffffff'))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(df, x='attendance_rate', y='risk_probability', color='risk_level',
            color_discrete_map={'Low Risk':'#10b981','Medium Risk':'#f59e0b','High Risk':'#ef4444'}, height=350)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,31,58,0.3)', font=dict(color='#ffffff'))
        st.plotly_chart(fig, use_container_width=True)

with t4:
    st.subheader("🧠 Model Insights & Explainability")
    pkg = load_model()
    X = df[FEAT].copy().astype(float)
    X_scaled = pkg['scaler'].transform(X)
    y_true = (df['risk_probability'] >= 0.5).astype(int)
    y_pred = pkg['model'].predict(X_scaled)
    y_pred_proba = pkg['model'].predict_proba(X_scaled)[:, 1]
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("#### Feature Importance")
        model = pkg['model']
        if hasattr(model,'feature_importances_'): imp = model.feature_importances_
        elif hasattr(model,'coef_'): imp = np.abs(model.coef_[0])
        else: imp = np.ones(len(FEAT))
        fi = pd.DataFrame({'Feature':FEAT,'Imp':imp}).sort_values('Imp',ascending=True)
        fi['Pct'] = fi['Imp']/fi['Imp'].sum()*100
        colors = ['#ef4444' if v>fi['Pct'].mean() else '#3b82f6' for v in fi['Pct']]
        fig = go.Figure(go.Bar(y=fi['Feature'],x=fi['Pct'],orientation='h',marker=dict(color=colors),text=[f'{v:.1f}%' for v in fi['Pct']],textposition='outside'))
        fig.update_layout(height=400,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(26,31,58,0.3)',font=dict(color='#ffffff'),xaxis=dict(title='Importance %',color='#d1d5db'),yaxis=dict(color='#d1d5db'),margin=dict(l=150))
        st.plotly_chart(fig,use_container_width=True)
        top3 = fi.tail(3)['Feature'].tolist()[::-1]
        st.info(f"Top 3: **{', '.join(top3)}**")
    with col2:
        st.markdown("#### Confusion Matrix")
        tyt = pkg.get('test_y_true',None); typ = pkg.get('test_y_pred',None)
        cm = confusion_matrix(tyt,typ) if tyt is not None else confusion_matrix(y_true,y_pred)
        fig = go.Figure(go.Heatmap(z=cm,x=['Predicted Safe','Predicted At-Risk'],y=['Actually Safe','Actually At-Risk'],
            colorscale=[[0,'#1a1f3a'],[0.5,'#3b82f6'],[1,'#10b981']],text=cm,texttemplate='%{text}',textfont=dict(size=20,color='white'),showscale=False))
        fig.update_layout(height=400,paper_bgcolor='rgba(0,0,0,0)',font=dict(color='#ffffff'),margin=dict(l=120))
        st.plotly_chart(fig,use_container_width=True)
        tn,fp,fn,tp = cm.ravel()
        acc=(tp+tn)/(tp+tn+fp+fn); prec=tp/(tp+fp) if tp+fp>0 else 0; rec=tp/(tp+fn) if tp+fn>0 else 0
        c1,c2,c3 = st.columns(3)
        with c1: st.metric("Accuracy",f"{acc:.1%}")
        with c2: st.metric("Precision",f"{prec:.1%}")
        with c3: st.metric("Recall",f"{rec:.1%}")
    st.divider()
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("#### ROC Curve")
        fpr,tpr,_ = roc_curve(y_true,y_pred_proba); roc_auc = auc(fpr,tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',name=f'ROC (AUC={roc_auc:.3f})',line=dict(color='#3b82f6',width=3),fill='tozeroy',fillcolor='rgba(59,130,246,0.2)'))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Random',line=dict(color='#6b7280',width=2,dash='dash')))
        fig.update_layout(height=350,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(26,31,58,0.3)',font=dict(color='#ffffff'),xaxis=dict(title='FPR',color='#d1d5db'),yaxis=dict(title='TPR',color='#d1d5db'))
        st.plotly_chart(fig,use_container_width=True)
        qual = 'Excellent' if roc_auc>0.85 else 'Good' if roc_auc>0.75 else 'Fair'
        st.caption(f"AUC = {roc_auc:.3f}  ({qual})")
    with col2:
        st.markdown("#### Precision-Recall")
        pv,rv,_ = precision_recall_curve(y_true,y_pred_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rv,y=pv,mode='lines',line=dict(color='#10b981',width=3),fill='tozeroy',fillcolor='rgba(16,185,129,0.2)'))
        fig.update_layout(height=350,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(26,31,58,0.3)',font=dict(color='#ffffff'),xaxis=dict(title='Recall',color='#d1d5db'),yaxis=dict(title='Precision',color='#d1d5db'))
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Precision-Recall Curve")

with t5:
    st.subheader("⚠️ Alerts")
    hrs = df[df['risk_level']=='High Risk'].sort_values('risk_probability',ascending=False)
    if len(hrs)==0: st.success("✓ No high-risk students!")
    else:
        st.error(f"⚠️ {len(hrs)} need intervention")
        for _,r in hrs.iterrows():
            with st.expander(f"🔴 {r['student_name']} ({r['risk_probability']:.0%})"):
                c1,c2,c3 = st.columns(3)
                with c1: st.metric("CGPA",f"{r['cgpa']:.2f}")
                with c2: st.metric("Attendance",f"{r['attendance_rate']*100:.0f}%")
                with c3: st.metric("Incidents",int(r['behavioral_incidents']))

with t6:
    st.subheader("🔮 Quick Predict")
    with st.form("pred_form"):
        c1,c2,c3 = st.columns(3)
        with c1:
            p_year = st.selectbox("Year",[1,2,3,4],key="pred_year",format_func=lambda x:f"Year {x}")
            p_cgpa = st.slider("CGPA",0.0,10.0,7.0,0.1,key="pred_cgpa")
        with c2:
            p_att  = st.slider("Attendance %",0,100,88,key="pred_att")
            p_comp = st.slider("Completion %",0,100,80,key="pred_comp")
        with c3:
            p_inc  = st.number_input("Incidents",0,15,0,key="pred_inc")
            p_fail = st.number_input("Failures",0,5,0,key="pred_fail")
        if st.form_submit_button("Predict",use_container_width=True,type="primary"):
            pkg = load_model()
            inp = pd.DataFrame([{"year":float(p_year),"cgpa":float(p_cgpa),"previous_failures":float(p_fail),
                "assignment_completion":float(p_comp/100),"attendance_rate":float(p_att/100),
                "unexcused_absences":0.0,"behavioral_incidents":float(p_inc),"days_since_incident":365.0,
                "scholarship":0.0,"extracurricular":1.0,"hostel_resident":0.0,"part_time_job":0.0}])
            prob = float(pkg["model"].predict_proba(pkg["scaler"].transform(inp[FEAT]))[0][1])
            col  = "#ef4444" if prob>=0.65 else "#f59e0b" if prob>=0.35 else "#10b981"
            lvl  = "HIGH RISK" if prob>=0.65 else "MEDIUM RISK" if prob>=0.35 else "LOW RISK"
            st.markdown(f'''<div style="background:#1a1f3a;border:3px solid {col};border-radius:16px;
                padding:2rem;text-align:center;margin-top:1rem;box-shadow:0 0 40px {col}44;">
                <div style="font-size:3rem;font-weight:900;color:{col};">{prob:.0%}</div>
                <div style="font-size:1.3rem;color:{col};font-weight:700;">{lvl}</div>
            </div>''', unsafe_allow_html=True)

with t7:
    st.markdown('''
    <div style="background:linear-gradient(135deg,#1e3a5f,#2563eb);padding:1.5rem 2rem;
        border-radius:16px;margin-bottom:1.5rem;border:1px solid #3b82f6;">
        <h2 style="color:#ffffff;margin:0;">➕ Add New Student</h2>
        <p style="color:#93c5fd;margin:0.3rem 0 0;">Fill in details below — risk predicted instantly on submit</p>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">Student Info</div>', unsafe_allow_html=True)
    oi1, oi2 = st.columns(2)
    with oi1:
        sname = st.text_input("Full Name", placeholder="e.g. Arjun Sharma", key="t7_name")
    with oi2:
        syear = st.selectbox("Year", [1,2,3,4], key="t7_year", format_func=lambda x: f"Year {x}")
    oi3, oi4 = st.columns(2)
    with oi3:
        sschool = st.selectbox("School", ['VSST','VSOD','VSOL','JAGSOM','TSM'], key="t7_school")
    with oi4:
        program_map = {
            'VSST':   ['B.Tech CSE','B.Tech AI/ML'],
            'VSOD':   ['B.Des Fashion','B.Des Product','B.Des Communication'],
            'VSOL':   ['BA LLB','BBA LLB','LLB'],
            'JAGSOM': ['BBA','MBA'],
            'TSM':    ['B.Mus Performance','B.Mus Production','B.Mus Composition','B.Tech Sound Engineering']
        }
        sprogram = st.selectbox("Program", program_map.get(sschool, ['General']), key=f"t7_prog_{sschool}")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.form("add_student_form_v3"):

        st.markdown('<div class="section-card"><div class="section-title">Academic Performance</div>', unsafe_allow_html=True)
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            scgpa = st.slider("CGPA", 0.0, 10.0, 7.0, 0.1, key="t7_cgpa")
            cgpa_color = "#10b981" if scgpa>=7.0 else "#f59e0b" if scgpa>=5.5 else "#ef4444"
            cgpa_label = "Good Standing" if scgpa>=7.0 else "Needs Attention" if scgpa>=5.5 else "At Risk"
            st.markdown(
                f'<div style="background:{cgpa_color}22;border:1px solid {cgpa_color};'                f'border-radius:8px;padding:0.4rem 0.8rem;color:{cgpa_color};'                f'font-weight:600;font-size:0.85rem;text-align:center;">{cgpa_label}</div>',
                unsafe_allow_html=True)
        with ac2:
            sfail = st.selectbox("Previous Failures", [0,1,2,3,4,5], key="t7_fail",
                                 format_func=lambda x: f"{x} subject{'s' if x!=1 else ''}")
        with ac3:
            spending = st.radio("Assignments Pending?",
                                ["No","Yes - Minor (1-2)","Yes - Major (3+)"], key="t7_pending")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><div class="section-title">Attendance by Course</div>', unsafe_allow_html=True)
        st.caption("Enter course names separated by commas — sliders appear automatically")
        courses_input = st.text_input("Course Names",
            placeholder="e.g. Mathematics, Physics, Python, Design", key="t7_courses")
        course_att = []
        if courses_input.strip():
            course_list = [c.strip() for c in courses_input.split(",") if c.strip()][:10]
            st.markdown(f"**{len(course_list)} course{'s' if len(course_list)!=1 else ''} detected:**")
            att_cols = st.columns(3)
            for i, course in enumerate(course_list):
                with att_cols[i % 3]:
                    val = st.slider(course, 0, 100, 90, 1, key=f"t7_att_{i}")
                    col = "#10b981" if val>=85 else "#f59e0b" if val>=75 else "#ef4444"
                    lbl = "Good" if val>=85 else "Low" if val>=75 else "Critical"
                    st.markdown(
                        f'<div style="height:4px;border-radius:2px;background:{col};'                        f'margin-top:-10px;margin-bottom:4px;"></div>'                        f'<div style="font-size:0.75rem;color:{col};text-align:center;margin-bottom:8px;">{val}% — {lbl}</div>',
                        unsafe_allow_html=True)
                    course_att.append(val)
        else:
            st.info("Type course names above to generate attendance sliders")
            course_att = [90]
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><div class="section-title">Behaviour & Support</div>', unsafe_allow_html=True)
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            sinc = st.selectbox("Disciplinary Incidents", [0,1,2,3,4,5], key="t7_inc",
                                format_func=lambda x: f"{x} incident{'s' if x!=1 else ''}")
        with bc2:
            sdays = st.selectbox("Last Incident", [365,180,90,60,30,14,7,1], key="t7_days",
                                 format_func=lambda x: "No incident" if x==365 else f"{x} days ago" if x>=30 else f"{x} days ago (recent!)")
        with bc3:
            sref = st.selectbox("Counsellor Referrals", [0,1,2,3,4,5], key="t7_ref",
                                format_func=lambda x: "None" if x==0 else f"{x} referral{'s' if x!=1 else ''}")
        with bc4:
            st.markdown("<br>", unsafe_allow_html=True)
            sscholarship = st.checkbox("Has Scholarship", key="t7_scholarship")
            sextra       = st.checkbox("Extracurricular", value=True, key="t7_extra")
            shostel      = st.checkbox("Hostel Resident",  key="t7_hostel")
            sparttime    = st.checkbox("Part-Time Job",    key="t7_parttime")
        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("PREDICT RISK & ADD STUDENT", use_container_width=True, type="primary")

    if submitted:
        if not sname.strip():
            st.error("Please enter the student name.")
        else:
            with st.spinner("Analysing student profile..."):
                import time
                time.sleep(0.8)
                avg_att  = sum(course_att)/len(course_att)/100.0 if course_att else 0.90
                comp_map = {"No":0.95,"Yes - Minor (1-2)":0.65,"Yes - Major (3+)":0.30}
                scomp    = comp_map[spending]
                try:
                    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
                    cur  = conn.cursor()
                    cur.execute("SELECT MAX(student_id) FROM students")
                    new_id = (cur.fetchone()[0] or 0) + 1
                    dt = datetime.now().strftime('%Y-%m-%d')
                    cur.execute("INSERT INTO students (student_id,student_name,school,program,year,enrollment_date) VALUES (?,?,?,?,?,?)",
                        (new_id,sname,sschool,sprogram,syear,dt))
                    cur.execute("INSERT INTO academic_records (student_id,cgpa,previous_failures,assignment_completion) VALUES (?,?,?,?)",
                        (new_id,scgpa,sfail,scomp))
                    unexc = 0
                    for day in range(30):
                        p = np.random.random()<avg_att
                        s = "Present" if p else "Absent"
                        u = 0 if p else int(np.random.random()<0.6)
                        unexc += u
                        d = (datetime.now()-timedelta(days=29-day)).strftime('%Y-%m-%d')
                        cur.execute("INSERT INTO attendance (student_id,date,status,unexcused) VALUES (?,?,?,?)",(new_id,d,s,u))
                    cur.execute("INSERT INTO behavioral_incidents (student_id,total_incidents,days_since_last) VALUES (?,?,?)",(new_id,sinc,sdays))
                    cur.execute("INSERT INTO student_demographics (student_id,scholarship,extracurricular,hostel_resident,part_time_job) VALUES (?,?,?,?,?)",
                        (new_id,int(sscholarship),int(sextra),int(shostel),int(sparttime)))
                    rd = (datetime.now()-timedelta(days=np.random.randint(1,180))).strftime('%Y-%m-%d') if sref>0 else None
                    cur.execute("INSERT INTO counselor_referrals (student_id,total_referrals,last_referral_date) VALUES (?,?,?)",(new_id,sref,rd))
                    conn.commit(); conn.close()

                    pkg   = load_model()
                    feats = pd.DataFrame([{
                        "year":float(syear),"cgpa":float(scgpa),"previous_failures":float(sfail),
                        "assignment_completion":float(scomp),"attendance_rate":float(avg_att),
                        "unexcused_absences":float(unexc),"behavioral_incidents":float(sinc),
                        "days_since_incident":float(sdays),"scholarship":float(sscholarship),
                        "extracurricular":float(sextra),"hostel_resident":float(shostel),"part_time_job":float(sparttime)
                    }])
                    X_sc = pkg['scaler'].transform(feats[FEAT])
                    prob = float(pkg['model'].predict_proba(X_sc)[0][1])

                    if prob>=0.65:   lvl,card_class,score_color,icon,desc = "HIGH RISK",  "result-card-high",  "#ef4444","🔴","This student needs immediate counsellor attention."
                    elif prob>=0.35: lvl,card_class,score_color,icon,desc = "MEDIUM RISK","result-card-medium","#f59e0b","🔶","Schedule a check-in within the next two weeks."
                    else:            lvl,card_class,score_color,icon,desc = "LOW RISK",   "result-card-low",   "#10b981","🟢","This student appears to be doing well."

                    factors = []
                    if scgpa<5.5:    factors.append("Low CGPA")
                    if avg_att<0.85: factors.append("Poor Attendance")
                    if scomp<0.65:   factors.append("Pending Assignments")
                    if sinc>=1:      factors.append("Disciplinary Incidents")
                    if sfail>=1:     factors.append("Previous Failures")
                    if sref>=1:      factors.append("Prior Counselling")
                    pills = "".join([f'<span class="factor-pill">{f}</span>' for f in factors]) if factors else '<span class="factor-pill">No major risk factors</span>'

                    st.markdown(f'''
                    <div style="background:rgba(0,0,0,0.6);border-radius:20px;padding:0.5rem;margin-top:1rem;">
                        <div class="{card_class}" style="margin:auto;">
                            <div style="font-size:3rem;">{icon}</div>
                            <div class="result-label" style="color:{score_color};">{lvl}</div>
                            <div class="result-score" style="color:{score_color};">{prob:.0%}</div>
                            <div style="color:#9ca3af;font-size:0.9rem;margin-bottom:1rem;">
                                Risk Score for <strong style="color:#ffffff;">{sname}</strong> | ID: {new_id}
                            </div>
                            <div>{pills}</div>
                            <div class="result-desc">{desc}</div>
                        </div>
                    </div>''', unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    m1,m2,m3,m4 = st.columns(4)
                    with m1: st.metric("CGPA",          f"{scgpa:.1f}")
                    with m2: st.metric("Avg Attendance", f"{avg_att*100:.0f}%")
                    with m3: st.metric("Incidents",      sinc)
                    with m4: st.metric("Risk Score",     f"{prob:.0%}")
                    st.cache_data.clear()

                except Exception as e:
                    st.error(f"Error adding student: {str(e)}")
