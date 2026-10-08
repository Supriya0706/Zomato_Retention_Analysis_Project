import React, { useState, useEffect } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, Area, ComposedChart
} from 'recharts';
import { UserCheck, UserX, Activity, PieChart as PieChartIcon, Star, TrendingUp, Zap, Sparkles, Users, LineChart as LineChartIcon } from 'lucide-react';

const API_URL = (import.meta.env.VITE_API_URL || '/api').replace(/\/$/, '');

function App() {
  const [stats, setStats] = useState(null);
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // --- Churn prediction form state ---
  const [predictForm, setPredictForm] = useState({ orders: 5, avg_rating: 3.5, last_order_days: 15 });
  const [prediction, setPrediction] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [predictError, setPredictError] = useState(null);

  const runPrediction = async () => {
    setPredicting(true);
    setPredictError(null);
    setPrediction(null);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          orders: Number(predictForm.orders),
          avg_rating: Number(predictForm.avg_rating),
          last_order_days: Number(predictForm.last_order_days),
        }),
      });
      if (!res.ok) throw new Error(`Predict endpoint failed: ${res.status}`);
      const result = await res.json();
      setPrediction(result);
    } catch (err) {
      setPredictError(err.message);
    } finally {
      setPredicting(false);
    }
  };

  const riskColor = (risk) => {
    if (risk === 'High') return 'var(--risk-high, #ef4444)';
    if (risk === 'Medium') return '#fbbf24';
    return '#10b981';
  };

  // --- K-Means segments ---
  const [segments, setSegments] = useState(null);
  const [segmentsLoading, setSegmentsLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_URL}/segments`)
      .then((res) => { if (!res.ok) throw new Error('segments failed'); return res.json(); })
      .then((d) => setSegments(d.summary))
      .catch((err) => console.error('Segments fetch error:', err))
      .finally(() => setSegmentsLoading(false));
  }, [API_URL]);

  const segmentColor = (seg) => {
    if (seg === 'Loyal') return '#10b981';
    if (seg === 'At-Risk') return '#fbbf24';
    return 'var(--risk-high, #ef4444)'; // Dormant
  };

  // --- SARIMA forecast ---
  const [forecastData, setForecastData] = useState(null);
  const [forecastLoading, setForecastLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_URL}/forecast`)
      .then((res) => { if (!res.ok) throw new Error('forecast failed'); return res.json(); })
      .then((d) => setForecastData(d))
      .catch((err) => console.error('Forecast fetch error:', err))
      .finally(() => setForecastLoading(false));
  }, [API_URL]);

  const forecastChartData = forecastData ? [
    ...forecastData.historical.map(h => ({ month: h.month, actual: h.total_orders })),
    ...forecastData.forecast.map(f => ({ month: f.month, forecast: f.forecast_orders, lower: f.lower_80, upper: f.upper_80 })),
  ] : [];

  useEffect(() => {
    setIsLoading(true);
    Promise.all([
      fetch(`${API_URL}/stats`).then(res => {
        if (!res.ok) throw new Error(`Stats endpoint failed: ${res.status}`);
        return res.json();
      }),
      fetch(`${API_URL}/data`).then(res => {
        if (!res.ok) throw new Error(`Data endpoint failed: ${res.status}`);
        return res.json();
      })
    ])
    .then(([statsData, dataRes]) => {
      setStats(statsData);
      setData(dataRes);
      setIsLoading(false);
      setError(null);
    })
    .catch(err => {
      console.error("Error fetching data:", err);
      setError(err.message);
      setIsLoading(false);
    });
  }, [API_URL]);

  const pieData = stats ? [
    { name: 'Retained', value: stats.retained, fill: '#10b981' },
    { name: 'Churned', value: stats.churned, fill: '#ef4444' }
  ] : [];

  const chartData = data ? 
    ['Low', 'Medium', 'High'].map(segment => {
      const segmentData = data.filter(d => d.order_segment === segment);
      return {
        name: `${segment} Volume`,
        Churned: segmentData.filter(d => d.churn === 1).length,
        Retained: segmentData.filter(d => d.churn === 0).length
      };
    }) : [];

  return (
    <div className="app-container">
      <header className="header" style={{ marginBottom: '2rem' }}>
        <h1 style={{ fontSize: '2.5rem', fontWeight: 800, color: 'var(--text-primary)' }}>Zomato Retention Analysis</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Live Analytics Hub • Powered by React & FastAPI</p>
      </header>

      {isLoading ? (
        <div style={{ textAlign: 'center', marginTop: '4rem' }}>
          <Activity size={48} className="animate-spin" style={{ color: 'var(--accent-primary)', margin: '0 auto' }} />
          <p style={{ marginTop: '1rem', color: 'var(--text-secondary)' }}>Syncing Analytics Gateway...</p>
        </div>
      ) : error ? (
        <div style={{ textAlign: 'center', marginTop: '4rem', padding: '2rem' }} className="glass-panel">
          <Activity size={48} style={{ color: 'var(--risk-high)', margin: '0 auto' }} />
          <h2 style={{ marginTop: '1rem' }}>Connection Error</h2>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>{error}</p>
          <div style={{ fontSize: '0.8rem', background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px', display: 'inline-block' }}>
            Attempted URL: <code>{API_URL}</code>
          </div>
          <p style={{ marginTop: '1rem' }}>Please verify that your backend is running and the <code>VITE_API_URL</code> environment variable is set correctly.</p>
        </div>
      ) : (
        <>
          <div className="stats-grid">
            <div className="glass-panel stat-card">
              <div className="value">{stats?.total_users?.toLocaleString() || 0}</div>
              <div className="label">Total Users</div>
            </div>
            <div className="glass-panel stat-card">
              <div className="value" style={{ color: 'var(--risk-high)' }}>{stats?.churn_rate || 0}%</div>
              <div className="label">Churn Rate</div>
            </div>
            <div className="glass-panel stat-card">
              <div className="value" style={{ color: '#fbbf24' }}>{stats?.avg_rating || 0} ★</div>
              <div className="label">Avg Rating</div>
            </div>
            <div className="glass-panel stat-card">
              <div className="value" style={{ color: '#3b82f6' }}>{stats?.avg_orders || 0}</div>
              <div className="label">Avg Orders</div>
            </div>
          </div>

          <div className="main-grid" style={{ gridTemplateColumns: '1fr' }}>
            <div className="glass-panel" style={{ padding: '2rem' }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '2rem' }}>
                <div>
                  <h3 className="chart-title"><PieChartIcon size={20} /> Overall Distribution</h3>
                  <div style={{ height: '300px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie data={pieData} cx="50%" cy="50%" innerRadius={80} outerRadius={100} paddingAngle={5} dataKey="value">
                          {pieData.map((entry, index) => <Cell key={`cell-${index}`} fill={entry.fill} />)}
                        </Pie>
                        <Tooltip contentStyle={{ backgroundColor: 'var(--bg-card)', borderColor: 'var(--glass-border)' }} />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <div>
                  <h3 className="chart-title"><TrendingUp size={20} /> Retention by Order Volume</h3>
                  <div style={{ height: '300px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                        <XAxis dataKey="name" stroke="var(--text-muted)" />
                        <YAxis stroke="var(--text-muted)" />
                        <Tooltip contentStyle={{ backgroundColor: 'var(--bg-card)', borderColor: 'var(--glass-border)' }} />
                        <Legend />
                        <Bar dataKey="Retained" stackId="a" fill="#10b981" radius={[0, 0, 4, 4]} />
                        <Bar dataKey="Churned" stackId="a" fill="#ef4444" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* SARIMA Order Volume Forecast */}
          <div style={{ marginTop: '2rem' }}>
            <div className="glass-panel" style={{ padding: '2rem' }}>
              <h3 className="chart-title"><LineChartIcon size={20} /> Order Volume Forecast (SARIMA)</h3>
              <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem', fontSize: '0.9rem' }}>
                6-month forecast with 80% confidence interval, backtested against naive baselines.
              </p>
              {forecastLoading ? (
                <p style={{ color: 'var(--text-muted)' }}>Loading forecast...</p>
              ) : !forecastData ? (
                <p style={{ color: 'var(--text-muted)' }}>Forecast unavailable.</p>
              ) : (
                <>
                  <div style={{ height: '320px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={forecastChartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="month" stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
                        <YAxis stroke="var(--text-muted)" domain={['auto', 'auto']} />
                        <Tooltip contentStyle={{ backgroundColor: 'var(--bg-card)', borderColor: 'var(--glass-border)' }} />
                        <Legend />
                        <Area type="monotone" dataKey="upper" stroke="none" fill="#ef4444" fillOpacity={0.1} name="80% CI upper" />
                        <Area type="monotone" dataKey="lower" stroke="none" fill="#ef4444" fillOpacity={0.1} name="80% CI lower" />
                        <Line type="monotone" dataKey="actual" stroke="#3b82f6" strokeWidth={2} dot={{ r: 2 }} name="Actual" connectNulls />
                        <Line type="monotone" dataKey="forecast" stroke="#ef4444" strokeWidth={2} dot={{ r: 2 }} name="SARIMA Forecast" connectNulls />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                  <div style={{ display: 'flex', gap: '1.5rem', marginTop: '1rem', flexWrap: 'wrap', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    {Object.entries(forecastData.backtest).map(([name, m]) => (
                      <div key={name} style={{ background: 'rgba(255,255,255,0.03)', padding: '0.6rem 1rem', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
                        <strong style={{ color: 'var(--text-primary)' }}>{name.replace(/_/g, ' ')}</strong>: RMSE {m.rmse}, MAE {m.mae}
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>

          {/* K-Means Customer Segments (Loyal / At-Risk / Dormant) */}
          <div style={{ marginTop: '2rem' }}>
            <div className="glass-panel" style={{ padding: '2rem' }}>
              <h3 className="chart-title"><Users size={20} /> Customer Segments (K-Means on RFM)</h3>
              <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem', fontSize: '0.9rem' }}>
                Unsupervised clustering on Recency, Frequency &amp; Monetary value — churn was not used as an input,
                yet segment churn rates line up with the Random Forest's own predictions.
              </p>
              {segmentsLoading ? (
                <p style={{ color: 'var(--text-muted)' }}>Loading segments...</p>
              ) : !segments ? (
                <p style={{ color: 'var(--text-muted)' }}>Segments unavailable.</p>
              ) : (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
                  {segments.map((s) => (
                    <div key={s.segment} style={{ background: 'rgba(255,255,255,0.03)', padding: '1.2rem', borderRadius: '12px', border: `1px solid ${segmentColor(s.segment)}` }}>
                      <div style={{ fontWeight: 700, color: segmentColor(s.segment), marginBottom: '0.5rem' }}>{s.segment}</div>
                      <div style={{ fontSize: '1.4rem', fontWeight: 700 }}>{s.user_count} users</div>
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.5rem', lineHeight: 1.6 }}>
                        Avg orders: {s.orders}<br />
                        Avg spend: ₹{s.total_spend}<br />
                        Avg days since order: {s.last_order_days}<br />
                        Churn rate: <span style={{ color: segmentColor(s.segment), fontWeight: 600 }}>{Math.round(s.churn * 100)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Live Churn Prediction — calls the trained Random Forest via /predict */}
          <div style={{ marginTop: '2rem' }}>
            <div className="glass-panel" style={{ padding: '2rem' }}>
              <h3 className="chart-title"><Sparkles size={20} /> Predict Churn for a User</h3>
              <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem', fontSize: '0.9rem' }}>
                Enter behavior for any user and get a live churn probability from the trained Random Forest model.
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
                <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Orders placed
                  <input
                    type="number" min="0" value={predictForm.orders}
                    onChange={(e) => setPredictForm({ ...predictForm, orders: e.target.value })}
                    style={{ display: 'block', width: '100%', marginTop: '0.4rem', padding: '0.5rem', borderRadius: '8px', border: '1px solid var(--glass-border)', background: 'rgba(255,255,255,0.03)', color: 'inherit' }}
                  />
                </label>
                <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Avg rating (1-5)
                  <input
                    type="number" min="1" max="5" step="0.1" value={predictForm.avg_rating}
                    onChange={(e) => setPredictForm({ ...predictForm, avg_rating: e.target.value })}
                    style={{ display: 'block', width: '100%', marginTop: '0.4rem', padding: '0.5rem', borderRadius: '8px', border: '1px solid var(--glass-border)', background: 'rgba(255,255,255,0.03)', color: 'inherit' }}
                  />
                </label>
                <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Days since last order
                  <input
                    type="number" min="0" value={predictForm.last_order_days}
                    onChange={(e) => setPredictForm({ ...predictForm, last_order_days: e.target.value })}
                    style={{ display: 'block', width: '100%', marginTop: '0.4rem', padding: '0.5rem', borderRadius: '8px', border: '1px solid var(--glass-border)', background: 'rgba(255,255,255,0.03)', color: 'inherit' }}
                  />
                </label>
              </div>
              <button
                onClick={runPrediction}
                disabled={predicting}
                style={{ padding: '0.7rem 1.5rem', borderRadius: '8px', border: 'none', background: 'var(--accent-primary, #ef4444)', color: '#fff', fontWeight: 600, cursor: predicting ? 'not-allowed' : 'pointer', opacity: predicting ? 0.6 : 1 }}
              >
                {predicting ? 'Predicting...' : 'Predict Churn Risk'}
              </button>

              {predictError && (
                <p style={{ color: 'var(--risk-high, #ef4444)', marginTop: '1rem' }}>{predictError}</p>
              )}

              {prediction && (
                <div style={{ marginTop: '1.5rem', display: 'flex', gap: '2rem', flexWrap: 'wrap', alignItems: 'center' }}>
                  <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1.2rem 1.5rem', borderRadius: '12px', border: `1px solid ${riskColor(prediction.risk_level)}` }}>
                    <div style={{ fontSize: '1.8rem', fontWeight: 700, color: riskColor(prediction.risk_level) }}>
                      {prediction.churn_probability}%
                    </div>
                    <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)' }}>Churn Probability</div>
                  </div>
                  <div>
                    <div style={{ fontWeight: 600 }}>
                      {prediction.prediction_label === 'Churned' ? <UserX size={18} style={{ verticalAlign: 'middle', marginRight: '6px', color: riskColor(prediction.risk_level) }} /> : <UserCheck size={18} style={{ verticalAlign: 'middle', marginRight: '6px', color: riskColor(prediction.risk_level) }} />}
                      {prediction.prediction_label} &middot; {prediction.risk_level} Risk
                    </div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.3rem' }}>
                      From the Random Forest classifier (77% test accuracy, 0.84 ROC-AUC)
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Dataset snapshot */}
          <div style={{ marginTop: '2rem' }}>
            <div className="glass-panel" style={{ textAlign: 'center', padding: '3rem' }}>
              <Zap size={48} style={{ color: 'var(--accent-primary)', marginBottom: '1rem' }} />
              <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>Dataset Snapshot</h2>
              <p style={{ color: 'var(--text-secondary)', maxWidth: '600px', margin: '0 auto 2rem' }}>
                All figures on this dashboard come from a synthetic dataset of 1,000 users generated for this project,
                in which churn depends on order count, rating and recency. It is not real Zomato data.
                A companion Power BI report is built on the same star-schema exports.
              </p>
              <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem', flexWrap: 'wrap' }}>
                <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1.5rem', borderRadius: '12px', border: '1px solid var(--glass-border)', minWidth: '150px' }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{stats?.retained || 0}</div>
                  <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)' }}>Retained users</div>
                </div>
                <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1.5rem', borderRadius: '12px', border: '1px solid var(--glass-border)', minWidth: '150px' }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{stats?.churned || 0}</div>
                  <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)' }}>Churned users</div>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default App;
