import React, { useState, useEffect } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell, ScatterChart, Scatter
} from 'recharts';
import { AlertCircle, UserCheck, UserX, Activity, PieChart as PieChartIcon } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [stats, setStats] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  // Predictor state
  const [formData, setFormData] = useState({
    orders: 5,
    avg_rating: 4.0,
    last_order_days: 14
  });
  const [prediction, setPrediction] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);

  useEffect(() => {
    Promise.all([
      fetch(`${API_URL}/stats`).then(res => res.json()),
      fetch(`${API_URL}/model-metrics`).then(res => res.json())
    ])
    .then(([statsData, metricsData]) => {
      setStats(statsData);
      setMetrics(metricsData);
      setIsLoading(false);
    })
    .catch(err => {
      console.error("Error fetching data:", err);
      setIsLoading(false);
    });
  }, []);

  const handlePredict = async (e) => {
    e.preventDefault();
    setIsPredicting(true);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      const data = await res.json();
      setPrediction(data);
    } catch (err) {
      console.error("Prediction error:", err);
    } finally {
      setIsPredicting(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: parseFloat(value)
    }));
  };

  // Mock data for charts since the API for /data might be heavy to load every time
  // In a real scenario, this would come from the /data endpoint
  const pieData = stats ? [
    { name: 'Retained', value: stats.retained, fill: '#10b981' },
    { name: 'Churned', value: stats.churned, fill: '#ef4444' }
  ] : [];

  const featureImportance = metrics ? Object.entries(metrics.feature_importance)
    .map(([name, value]) => ({ name, importance: value }))
    .sort((a, b) => b.importance - a.importance)
  : [];

  return (
    <div className="app-container">
      <header className="header">
        <h1>Zomato Retention Analytics</h1>
        <p>Machine Learning powered churn prediction and insights</p>
      </header>

      {isLoading ? (
        <div style={{ textAlign: 'center', marginTop: '4rem' }}>
          <Activity size={48} className="animate-spin" style={{ color: 'var(--accent-primary)', margin: '0 auto' }} />
          <p style={{ marginTop: '1rem', color: 'var(--text-secondary)' }}>Loading analytics payload...</p>
        </div>
      ) : (
        <>
          {/* Top Stats Row */}
          <div className="stats-grid">
            <div className="glass-panel stat-card">
              <UserCheck size={24} style={{ color: 'var(--risk-low)', margin: '0 auto 0.5rem' }} />
              <div className="value">{stats?.total_users?.toLocaleString() || 0}</div>
              <div className="label">Total Users Analyzed</div>
            </div>
            <div className="glass-panel stat-card">
              <UserX size={24} style={{ color: 'var(--risk-high)', margin: '0 auto 0.5rem' }} />
              <div className="value">{stats?.churn_rate || 0}%</div>
              <div className="label">Historical Churn Rate</div>
            </div>
            <div className="glass-panel stat-card">
              <Activity size={24} style={{ color: 'var(--accent-primary)', margin: '0 auto 0.5rem' }} />
              <div className="value">{(metrics?.accuracy * 100)?.toFixed(1) || 0}%</div>
              <div className="label">Model Accuracy (RF)</div>
            </div>
            <div className="glass-panel stat-card">
              <PieChartIcon size={24} style={{ color: '#3b82f6', margin: '0 auto 0.5rem' }} />
              <div className="value">{stats?.avg_orders || 0}</div>
              <div className="label">Avg Orders / User</div>
            </div>
          </div>

          <div className="main-grid">
            {/* Left Column: Predictor */}
            <div className="glass-panel">
              <h2 className="chart-title">
                <AlertCircle size={20} />
                Live Predictor
              </h2>
              <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
                Enter user behavior to predict their churn likelihood based on our Random Forest model.
              </p>

              <form onSubmit={handlePredict}>
                <div className="input-group">
                  <label htmlFor="orders">Total Orders (0-100)</label>
                  <input
                    type="number"
                    id="orders"
                    name="orders"
                    className="input-field"
                    value={formData.orders}
                    onChange={handleInputChange}
                    min="0"
                    max="100"
                    required
                  />
                </div>
                
                <div className="input-group">
                  <label htmlFor="avg_rating">Avg Rating (1.0 - 5.0)</label>
                  <input
                    type="number"
                    id="avg_rating"
                    name="avg_rating"
                    className="input-field"
                    value={formData.avg_rating}
                    onChange={handleInputChange}
                    min="1"
                    max="5"
                    step="0.1"
                    required
                  />
                </div>
                
                <div className="input-group">
                  <label htmlFor="last_order_days">Days Since Last Order (0-365)</label>
                  <input
                    type="number"
                    id="last_order_days"
                    name="last_order_days"
                    className="input-field"
                    value={formData.last_order_days}
                    onChange={handleInputChange}
                    min="0"
                    max="365"
                    required
                  />
                </div>

                <button 
                  type="submit" 
                  className="btn-primary"
                  disabled={isPredicting}
                >
                  {isPredicting ? 'Analyzing...' : 'Predict Churn Risk'}
                </button>
              </form>

              {prediction && (
                <div className="prediction-result">
                  <div className={`risk-circle risk-${prediction.risk_level}`}>
                    <div className="prob">{prediction.churn_probability.toFixed(0)}%</div>
                    <div className="label">Risk</div>
                  </div>
                  <h3 style={{ marginBottom: '0.25rem' }}>
                    {prediction.prediction_label}
                  </h3>
                  <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                    Model indicates a {prediction.risk_level.toLowerCase()} risk of churn.
                  </p>
                </div>
              )}
            </div>

            {/* Right Column: Visualizations */}
            <div className="glass-panel" style={{ padding: '0' }}>
              <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--glass-border)' }}>
                <h2 className="chart-title" style={{ margin: 0 }}>Model Diagnostics & Analysis</h2>
              </div>
              
              <div className="charts-grid" style={{ padding: '1.5rem' }}>
                {/* Churn Distribution */}
                <div>
                  <h3 style={{ fontSize: '1rem', marginBottom: '1rem', color: 'var(--text-secondary)' }}>Overall Churn Distribution</h3>
                  <div className="chart-container" style={{ height: '250px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={pieData}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={80}
                          paddingAngle={5}
                          dataKey="value"
                        >
                          {pieData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.fill} />
                          ))}
                        </Pie>
                        <Tooltip 
                          contentStyle={{ backgroundColor: 'var(--bg-card)', borderColor: 'var(--glass-border)', color: '#fff' }}
                          itemStyle={{ color: '#fff' }}
                        />
                        <Legend verticalAlign="bottom" height={36}/>
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Feature Importance */}
                <div>
                  <h3 style={{ fontSize: '1rem', marginBottom: '1rem', color: 'var(--text-secondary)' }}>Feature Importance (RF)</h3>
                  <div className="chart-container" style={{ height: '250px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        layout="vertical"
                        data={featureImportance}
                        margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" horizontal={false} />
                        <XAxis type="number" stroke="var(--text-muted)" tickFormatter={(tick) => `${(tick * 100).toFixed(0)}%`} />
                        <YAxis dataKey="name" type="category" stroke="var(--text-muted)" fontSize={12} width={100} />
                        <Tooltip 
                          formatter={(value) => `${(value * 100).toFixed(1)}%`}
                          contentStyle={{ backgroundColor: 'var(--bg-card)', borderColor: 'var(--glass-border)', color: '#fff' }}
                        />
                        <Bar dataKey="importance" fill="var(--accent-primary)" radius={[0, 4, 4, 0]} barSize={24} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              {/* Metrics footer */}
              <div style={{ padding: '1rem 1.5rem', background: 'rgba(0,0,0,0.2)', borderBottomLeftRadius: '16px', borderBottomRightRadius: '16px', display: 'flex', gap: '2rem', flexWrap: 'wrap', fontSize: '0.875rem' }}>
                <div><span style={{ color: 'var(--text-secondary)' }}>Precision:</span> {metrics?.precision}</div>
                <div><span style={{ color: 'var(--text-secondary)' }}>Recall:</span> {metrics?.recall}</div>
                <div><span style={{ color: 'var(--text-secondary)' }}>F1 Score:</span> {metrics?.f1_score}</div>
                <div><span style={{ color: 'var(--text-secondary)' }}>ROC-AUC:</span> {metrics?.roc_auc}</div>
              </div>
            </div>
          </div>

          {/* Power BI Section */}
          <div className="powerbi-section">
            <h2 className="header" style={{ marginBottom: '1rem' }}><span style={{ fontSize: '1.5rem', color: 'var(--text-primary)' }}>Live Power BI Embed</span></h2>
            <p style={{ textAlign: 'center', color: 'var(--text-secondary)', marginBottom: '2rem' }}>
              Interactive executive dashboard powered by Microsoft Power BI.
            </p>
            
            <div className="glass-panel" style={{ padding: '0.5rem' }}>
              <div className="iframe-container">
                {/* 
                  To actually embed, replace the URL below with your Power BI Publish to Web URL:
                  <iframe title="Zomato Analytics" src="YOUR_POWER_BI_URL" allowFullScreen></iframe> 
                */}
                <div className="powerbi-fallback">
                  <PieChartIcon size={64} style={{ opacity: 0.5 }} />
                  <h3 style={{ marginTop: '1rem', marginBottom: '0.5rem' }}>Power BI Dashboard Reserved Space</h3>
                  <p style={{ maxWidth: '500px', fontSize: '0.875rem' }}>
                    Follow the instructions in <code>powerbi/README_powerbi.md</code> to generate your Power BI report and paste the iframe embed code here (in <code>App.jsx</code>).
                  </p>
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
