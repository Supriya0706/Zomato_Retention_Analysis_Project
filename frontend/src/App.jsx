import React, { useState, useEffect } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell
} from 'recharts';
import { UserCheck, UserX, Activity, PieChart as PieChartIcon, Star } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
console.log(`[App] Using API URL: ${API_URL}`);

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

  // Data analysis aggregations for Charts
  const chartData = data ? 
    ['Low', 'Medium', 'High'].map(segment => {
      const segmentData = data.filter(d => d.order_segment === segment);
      const churned = segmentData.filter(d => d.churn === 1).length;
      const retained = segmentData.filter(d => d.churn === 0).length;
      return {
        name: `${segment} Orders`,
        Churned: churned,
        Retained: retained
      };
    }) : [];

  return (
    <div className="app-container">
      <header className="header">
        <h1>Zomato Retention Analytics</h1>
        <p>Data-driven insights into customer churn and retention</p>
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
              <Star size={24} style={{ color: '#fbbf24', margin: '0 auto 0.5rem' }} />
              <div className="value">{stats?.avg_rating || 0}</div>
              <div className="label">Avg User Rating</div>
            </div>
            <div className="glass-panel stat-card">
              <PieChartIcon size={24} style={{ color: '#3b82f6', margin: '0 auto 0.5rem' }} />
              <div className="value">{stats?.avg_orders || 0}</div>
              <div className="label">Avg Orders / User</div>
            </div>
          </div>

          <div className="main-grid" style={{ gridTemplateColumns: '1fr' }}>
            <div className="glass-panel" style={{ padding: '0' }}>
              <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--glass-border)' }}>
                <h2 className="chart-title" style={{ margin: 0 }}>Exploratory Data Analysis</h2>
              </div>
              
              <div className="charts-grid" style={{ padding: '1.5rem', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))' }}>
                {/* Churn Distribution */}
                <div>
                  <h3 style={{ fontSize: '1rem', marginBottom: '1rem', color: 'var(--text-secondary)' }}>Overall Churn Distribution</h3>
                  <div className="chart-container" style={{ height: '300px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={pieData}
                          cx="50%"
                          cy="50%"
                          innerRadius={80}
                          outerRadius={100}
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

                {/* Churn by Order Volume */}
                <div>
                  <h3 style={{ fontSize: '1rem', marginBottom: '1rem', color: 'var(--text-secondary)' }}>Retention by Order Volume</h3>
                  <div className="chart-container" style={{ height: '300px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={chartData}
                        margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                        <XAxis dataKey="name" stroke="var(--text-muted)" />
                        <YAxis stroke="var(--text-muted)" />
                        <Tooltip 
                          contentStyle={{ backgroundColor: 'var(--bg-card)', borderColor: 'var(--glass-border)', color: '#fff' }}
                        />
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

          {/* Power BI Section */}
          <div className="powerbi-section" style={{ marginTop: '3rem' }}>
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
