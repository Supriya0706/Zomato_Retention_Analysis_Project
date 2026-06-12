import React, { useState, useEffect } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell
} from 'recharts';
import { UserCheck, UserX, Activity, PieChart as PieChartIcon, Star, TrendingUp } from 'lucide-react';

const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '');

function App() {
  const [stats, setStats] = useState(null);
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch(`${API_URL}/stats`).then(res => res.json()),
      fetch(`${API_URL}/data`).then(res => res.json())
    ])
    .then(([statsData, dataRes]) => {
      setStats(statsData);
      setData(dataRes);
      setIsLoading(false);
    })
    .catch(err => {
      console.error("Error fetching data:", err);
      setIsLoading(false);
    });
  }, []);

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

          {/* New Clean Replacement for Power BI Section */}
          <div style={{ marginTop: '2rem' }}>
            <div className="glass-panel" style={{ textAlign: 'center', padding: '3rem' }}>
              <Zap size={48} style={{ color: 'var(--accent-primary)', marginBottom: '1rem' }} />
              <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>Automated Retention Monitoring</h2>
              <p style={{ color: 'var(--text-secondary)', maxWidth: '600px', margin: '0 auto 2rem' }}>
                This section replaces the static Power BI report with a live, synchronized analytics feed. 
                Data is ingested and processed through the Star Schema pipeline to ensure absolute metric accuracy.
              </p>
              <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem', flexWrap: 'wrap' }}>
                <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1.5rem', borderRadius: '12px', border: '1px solid var(--glass-border)', minWidth: '150px' }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{stats?.retained || 0}</div>
                  <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)' }}>Healthy Base</div>
                </div>
                <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1.5rem', borderRadius: '12px', border: '1px solid var(--glass-border)', minWidth: '150px' }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{stats?.churned || 0}</div>
                  <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)' }}>Churn Risk</div>
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
