import React, { useState, useEffect, useRef } from 'react';
import { Send, BarChart3, TrendingUp, Users, DollarSign, Search, Loader2, RefreshCw, PieChart as PieChartIcon, LineChart as LineChartIcon, Table, AlertTriangle, CheckCircle, Activity, Target, FileText, Database, Download, FileDown, Clock } from 'lucide-react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

// Import separated CSS files
import '../styles/Dashboard.css';
import '../styles/Visualizations.css';
import '../styles/Animations.css';
import '../styles/Typography.css';
import '../styles/Responsive.css';

const API_BASE_URL = 'http://localhost:8083';

// Enhanced color palette for charts
const COLORS = ['#8B5CF6', '#06B6D4', '#10B981', '#F59E0B', '#EF4444', '#8B5A2B', '#6366F1', '#EC4899', '#84CC16', '#F97316'];

// Value formatting utilities
const ValueFormatter = {
  currency: (value) => {
    if (typeof value !== 'number') return value;
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  },

  percentage: (value) => {
    if (typeof value !== 'number') return value;
    return `${value.toFixed(1)}%`;
  },

  number: (value) => {
    if (typeof value !== 'number') return value;
    return value.toLocaleString();
  },

  formatValue: (value, format) => {
    switch (format) {
      case 'currency': return ValueFormatter.currency(value);
      case 'percentage': return ValueFormatter.percentage(value);
      case 'number': return ValueFormatter.number(value);
      default: return value;
    }
  }
};

// Streaming Status Component
const StreamingStatus = ({ status, message, iteration, currentFunction }) => {
  const getStatusIcon = () => {
    switch (status) {
      case 'started':
        return <Clock size={16} className="text-blue-500 animate-pulse" />;
      case 'processing':
        return <Loader2 size={16} className="text-purple-500 animate-spin" />;
      case 'function_call':
        return <Activity size={16} className="text-orange-500 animate-pulse" />;
      case 'function_complete':
        return <CheckCircle size={16} className="text-green-500" />;
      case 'creating_visualizations':
        return <BarChart3 size={16} className="text-blue-500 animate-pulse" />;
      case 'generating_response':
        return <FileText size={16} className="text-purple-500 animate-pulse" />;
      case 'complete':
        return <CheckCircle size={16} className="text-green-500" />;
      case 'error':
        return <AlertTriangle size={16} className="text-red-500" />;
      default:
        return <Loader2 size={16} className="animate-spin" />;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'started':
      case 'creating_visualizations':
        return 'bg-blue-50 border-blue-200 text-blue-700';
      case 'processing':
      case 'generating_response':
        return 'bg-purple-50 border-purple-200 text-purple-700';
      case 'function_call':
        return 'bg-orange-50 border-orange-200 text-orange-700';
      case 'function_complete':
      case 'complete':
        return 'bg-green-50 border-green-200 text-green-700';
      case 'error':
        return 'bg-red-50 border-red-200 text-red-700';
      default:
        return 'bg-gray-50 border-gray-200 text-gray-700';
    }
  };

  return (
    <div className={`streaming-status ${getStatusColor()}`}>
      <div className="status-content">
        <div className="status-icon">
          {getStatusIcon()}
        </div>
        <div className="status-text">
          <span className="status-message">{message}</span>
          {iteration && (
            <span className="status-iteration">Step {iteration}</span>
          )}
          {currentFunction && (
            <span className="status-function">Running: {currentFunction}</span>
          )}
        </div>
      </div>
    </div>
  );
};

// PDF Download Component
const PDFDownloadButton = ({ conversationId, disabled, onDownloadStart, onDownloadComplete }) => {
  const [isDownloading, setIsDownloading] = useState(false);

  const handleDownload = async () => {
    if (!conversationId || isDownloading) return;

    setIsDownloading(true);
    onDownloadStart?.();

    try {
      const response = await fetch(`${API_BASE_URL}/api/report/pdf`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          conversation_id: conversationId,
          title: 'Revenue Analytics Report',
          include_visualizations: true,
          include_insights: true
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Get the blob data
      const blob = await response.blob();
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      
      // Generate filename with timestamp
      const timestamp = new Date().toISOString().slice(0, 19).replace(/[-:]/g, '').replace('T', '_');
      link.download = `revenue_analytics_report_${timestamp}.pdf`;
      
      // Trigger download
      document.body.appendChild(link);
      link.click();
      
      // Cleanup
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
      onDownloadComplete?.(true);
      
    } catch (error) {
      console.error('PDF download failed:', error);
      onDownloadComplete?.(false, error.message);
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <button
      onClick={handleDownload}
      disabled={disabled || isDownloading || !conversationId}
      className="pdf-download-button"
      title="Download as PDF Report"
    >
      {isDownloading ? (
        <>
          <Loader2 size={14} className="animate-spin" />
          <span>Generating PDF...</span>
        </>
      ) : (
        <>
          <FileDown size={14} />
          <span>Download Report PDF</span>
        </>
      )}
    </button>
  );
};

// Data Sources Citation Component
const DataSourcesCitation = ({ dataSources, className = "" }) => {
  if (!dataSources || dataSources.length === 0) return null;

  return (
    <div className={`data-sources-citation ${className}`}>
      <Database size={12} />
      <span className="sources-text">
        Source{dataSources.length > 1 ? 's' : ''}: {dataSources.join(', ')}
      </span>
    </div>
  );
};

// Key Metrics Dashboard Component
const KeyMetricsDashboard = ({ metrics }) => {
  if (!metrics || Object.keys(metrics).length === 0) return null;

  const getIcon = (label) => {
    const labelLower = label.toLowerCase();
    if (labelLower.includes('pipeline') || labelLower.includes('revenue') || labelLower.includes('value')) {
      return <DollarSign size={20} />;
    }
    if (labelLower.includes('rate') || labelLower.includes('percentage')) {
      return <Target size={20} />;
    }
    if (labelLower.includes('deal') || labelLower.includes('count')) {
      return <Activity size={20} />;
    }
    return <TrendingUp size={20} />;
  };

  const getTrendIcon = (trend) => {
    switch (trend) {
      case 'up': return <TrendingUp size={14} className="trend-up" />;
      case 'down': return <TrendingUp size={14} className="trend-down" />;
      default: return <div className="trend-stable">●</div>;
    }
  };

  return (
    <div className="key-metrics-dashboard">
      <div className="metrics-grid">
        {Object.entries(metrics).map(([key, metric]) => (
          <div key={key} className="metric-card">
            <div className="metric-header">
              <div className="metric-icon">
                {getIcon(metric.label)}
              </div>
              <div className="metric-trend">
                {getTrendIcon(metric.trend)}
              </div>
            </div>
            <div className="metric-value">
              {ValueFormatter.formatValue(metric.value, metric.format)}
            </div>
            <div className="metric-label">{metric.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Smart Chart Component based on chart type
const SmartChart = ({ visualization }) => {
  const { chart_type, data, title, description, data_sources } = visualization;

  const renderChart = () => {
    switch (chart_type) {
      case 'metrics':
        return <MetricsChart data={data} />;
      case 'bar':
        return <BarChartView data={data} />;
      case 'pie':
        return <PieChartView data={data} />;
      case 'line':
        return <LineChartView data={data} />;
      case 'table':
        return <TableView data={data} />;
      default:
        return <div className="unsupported-chart">Unsupported chart type: {chart_type}</div>;
    }
  };

  return (
    <div className="smart-chart">
      <div className="chart-header">
        <div className="chart-title-row">
          <h4 className="chart-title">{title}</h4>
          <DataSourcesCitation dataSources={data_sources} className="chart-citation" />
        </div>
        <p className="chart-description">{description}</p>
      </div>
      <div className="chart-content">
        {renderChart()}
      </div>
    </div>
  );
};

// Metrics Chart for KPI display
const MetricsChart = ({ data }) => (
  <div className="metrics-chart">
    {data.map((metric, index) => (
      <div key={index} className="metric-item">
        <div className="metric-value-large">
          {ValueFormatter.formatValue(metric.value, metric.format)}
        </div>
        <div className="metric-label-large">{metric.label}</div>
      </div>
    ))}
  </div>
);

// Enhanced Bar Chart
const BarChartView = ({ data }) => {
  if (!data || data.length === 0) return <div>No data available</div>;

  const formatYAxisTick = (value) => {
    if (value >= 1000000) {
      return `${(value / 1000000).toFixed(1)}M`;
    } else if (value >= 1000) {
      return `${(value / 1000).toFixed(0)}K`;
    }
    return ValueFormatter.number(value);
  };

  const customTooltipFormatter = (value, name, props) => {
    const dataPoint = props.payload;
    if (dataPoint && dataPoint.formatted) {
      return [dataPoint.formatted, name];
    }
    return [ValueFormatter.number(value), name];
  };

  return (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={data} margin={{ top: 20, right: 30, left: 40, bottom: 60 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis 
          dataKey="name" 
          angle={-45}
          textAnchor="end"
          height={60}
          fontSize={12}
          stroke="#6b7280"
        />
        <YAxis 
          tickFormatter={formatYAxisTick}
          fontSize={12}
          stroke="#6b7280"
        />
        <Tooltip 
          formatter={customTooltipFormatter}
          labelFormatter={(label) => `Category: ${label}`}
          contentStyle={{
            backgroundColor: '#ffffff',
            border: '1px solid #e5e7eb',
            borderRadius: '8px',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
          }}
        />
        <Bar 
          dataKey="value" 
          fill={COLORS[0]}
          radius={[4, 4, 0, 0]}
        />
      </BarChart>
    </ResponsiveContainer>
  );
};

// Enhanced Pie Chart
const PieChartView = ({ data }) => {
  if (!data || data.length === 0) return <div>No data available</div>;

  const customPieTooltipFormatter = (value, name, props) => {
    const dataPoint = props.payload;
    if (dataPoint && dataPoint.formatted) {
      return [dataPoint.formatted, 'Value'];
    }
    return [ValueFormatter.number(value), 'Value'];
  };

  return (
    <ResponsiveContainer width="100%" height={400}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          labelLine={false}
          label={({ name, percent }) => 
            percent > 0.05 ? `${name}: ${(percent * 100).toFixed(1)}%` : ''
          }
          outerRadius={120}
          fill="#8884d8"
          dataKey="value"
          stroke="#ffffff"
          strokeWidth={2}
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip 
          formatter={customPieTooltipFormatter}
          labelFormatter={(label, payload) => 
            payload && payload[0] ? 
            `Category: ${payload[0].payload.name}` : 
            `Category: ${label}`
          }
          contentStyle={{
            backgroundColor: '#ffffff',
            border: '1px solid #e5e7eb',
            borderRadius: '8px',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
          }}
        />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  );
};

// Enhanced Line Chart
const LineChartView = ({ data }) => {
  if (!data || data.length === 0) return <div>No data available</div>;

  const numericKeys = Object.keys(data[0]).filter(key => 
    typeof data[0][key] === 'number' && key !== 'period'
  );

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={data} margin={{ top: 20, right: 30, left: 40, bottom: 60 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis 
          dataKey="period" 
          fontSize={12}
          stroke="#6b7280"
        />
        <YAxis 
          fontSize={12}
          stroke="#6b7280"
        />
        <Tooltip 
          contentStyle={{
            backgroundColor: '#ffffff',
            border: '1px solid #e5e7eb',
            borderRadius: '8px',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
          }}
        />
        <Legend />
        {numericKeys.map((key, idx) => (
          <Line 
            key={key}
            type="monotone" 
            dataKey={key} 
            stroke={COLORS[idx % COLORS.length]}
            strokeWidth={3}
            dot={{ fill: COLORS[idx % COLORS.length], strokeWidth: 2, r: 4 }}
            activeDot={{ r: 6 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
};

// Enhanced Table View
const TableView = ({ data }) => {
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  const sortedData = React.useMemo(() => {
    if (!data || data.length === 0 || !sortConfig.key) return data || [];
    
    return [...data].sort((a, b) => {
      const aVal = a[sortConfig.key];
      const bVal = b[sortConfig.key];
      
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortConfig.direction === 'asc' ? aVal - bVal : bVal - aVal;
      }
      
      const aStr = String(aVal).toLowerCase();
      const bStr = String(bVal).toLowerCase();
      
      if (sortConfig.direction === 'asc') {
        return aStr < bStr ? -1 : aStr > bStr ? 1 : 0;
      } else {
        return aStr > bStr ? -1 : aStr < bStr ? 1 : 0;
      }
    });
  }, [data, sortConfig]);

  if (!data || data.length === 0) return <div>No data available</div>;

  const paginatedData = sortedData.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  const totalPages = Math.ceil(data.length / itemsPerPage);

  const handleSort = (key) => {
    setSortConfig(current => ({
      key,
      direction: current.key === key && current.direction === 'asc' ? 'desc' : 'asc'
    }));
  };

  const columns = Object.keys(data[0]);

  return (
    <div className="table-container">
      <div className="table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              {columns.map(column => (
                <th 
                  key={column}
                  onClick={() => handleSort(column)}
                  className={`sortable ${sortConfig.key === column ? sortConfig.direction : ''}`}
                >
                  <div className="th-content">
                    <span>{column.replace(/_/g, ' ').toUpperCase()}</span>
                    <div className="sort-indicator">
                      {sortConfig.key === column && (
                        <span>{sortConfig.direction === 'asc' ? '↑' : '↓'}</span>
                      )}
                    </div>
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paginatedData.map((row, idx) => (
              <tr key={idx}>
                {columns.map(column => (
                  <td key={column}>
                    {typeof row[column] === 'number' ? 
                      ValueFormatter.number(row[column]) : 
                      String(row[column] || '')
                    }
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {totalPages > 1 && (
        <div className="pagination">
          <button 
            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
            disabled={currentPage === 1}
            className="pagination-btn"
          >
            Previous
          </button>
          <span className="pagination-info">
            Page {currentPage} of {totalPages} ({data.length} total items)
          </span>
          <button 
            onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
            disabled={currentPage === totalPages}
            className="pagination-btn"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
};

// Visualizations Container
const VisualizationsContainer = ({ visualizations, keyMetrics, dataSources }) => {
  if (!visualizations && !keyMetrics) return null;

  return (
    <div className="visualizations-container">
      {/* Key Metrics Dashboard */}
      {keyMetrics && Object.keys(keyMetrics).length > 0 && (
        <div className="section">
          <div className="section-header">
            <h3 className="section-title">Key Metrics</h3>
            <DataSourcesCitation dataSources={dataSources} className="section-citation" />
          </div>
          <KeyMetricsDashboard metrics={keyMetrics} />
        </div>
      )}

      {/* Individual Visualizations */}
      {visualizations && visualizations.length > 0 && (
        <div className="section">
          <h3 className="section-title">
            Detailed Analysis ({visualizations.length} visualization{visualizations.length > 1 ? 's' : ''})
          </h3>
          <div className="visualizations-grid">
            {visualizations.map((viz, index) => (
              <div key={index} className="visualization-card">
                <SmartChart visualization={viz} />
                {viz.key_insights && viz.key_insights.length > 0 && (
                  <div className="insights-section">
                    <h5>Key Insights</h5>
                    <ul className="insights-list">
                      {viz.key_insights.map((insight, idx) => (
                        <li key={idx}>{insight}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// Enhanced message content formatter
const formatMessageContent = (content) => {
  if (!content) return content;
  
  let formatted = content
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/\n/g, '<br>')
    .replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>')
    .replace(/`(.*?)`/g, '<code>$1</code>')
    .replace(/^#### (.*$)/gim, '<h4>$1</h4>')
    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
    .replace(/^# (.*$)/gim, '<h1>$1</h1>')
    .replace(/^\* (.*$)/gim, '<li>$1</li>')
    .replace(/^- (.*$)/gim, '<li>$1</li>')
    .replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>')
    .replace(/<\/ul>\s*<ul>/g, '');
  
  return formatted;
};

// System Status Component
const SystemStatusIndicator = ({ systemStatus, onRefresh }) => (
  <div className="system-status">
    <div className={`status-dot ${systemStatus?.status === 'healthy' && systemStatus?.data_loaded ? 'healthy' : 'error'}`} />
    <span className="status-text">
      {systemStatus?.data_loaded 
        ? `${systemStatus?.records_count?.opportunities || 0} opportunities, ${systemStatus?.records_count?.contacts || 0} contacts`
        : 'System not ready'
      }
    </span>
    <button onClick={onRefresh} className="refresh-btn">
      <RefreshCw size={12} />
    </button>
  </div>
);

// context indicator component
const ContextIndicator = ({ hasContext }) => {
  if (!hasContext) return null;
  
  return (
    <div className="context-indicator">
      <div className="context-dot" />
      <span className="context-text">Maintaining conversation context</span>
    </div>
  );
};

// Main Dashboard Component
const RevenueAnalyticsDashboard = () => {
  const [query, setQuery] = useState('');
  const [conversations, setConversations] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingStatus, setStreamingStatus] = useState(null);
  const [systemStatus, setSystemStatus] = useState(null);
  const [activeTab, setActiveTab] = useState('chat');
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [error, setError] = useState(null);
  const chatEndRef = useRef(null);
  const eventSourceRef = useRef(null);

  // Enhanced quick questions
  const quickQuestions = [
    "What are the top 5 accounts by total opportunity value?",
    "Show me our pipeline breakdown by stage with metrics",
    "Which contact titles appear most in winning deals?", 
    "What's our win rate by opportunity type?",
    "Which products have the highest conversion rates?",
    "What's our expected revenue from high probability deals?",
    "What's the average performance compared across different dimensions?",
    "Show me missing stakeholders in high-value deals",
    "What's our comprehensive pipeline health analysis?"
  ];

  useEffect(() => {
    checkSystemHealth();
    scrollToBottom();
  }, [conversations]);

  useEffect(() => {
    // Cleanup event source on component unmount
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const checkSystemHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/health`);
      const data = await response.json();
      setSystemStatus(data);
      setError(null);
    } catch (error) {
      console.error('Health check failed:', error);
      setSystemStatus({ status: 'error', data_loaded: false });
      setError(`Cannot connect to backend server at ${API_BASE_URL}`);
    }
  };

  const handleStreamingQuery = async (questionText = query) => {
    if (!questionText.trim() || isStreaming) return;

    const userMessage = {
        id: Date.now(),
        type: 'user',
        content: questionText,
        timestamp: new Date()
    };

    setConversations(prev => [...prev, userMessage]);
    setIsStreaming(true);
    setIsLoading(true);
    setQuery('');
    setError(null);

    // Show a simple loading message
    setStreamingStatus({
        status: 'processing',
        message: 'Analyzing your request...',
        iteration: null,
        currentFunction: null
    });

    try {
        const response = await fetch(`${API_BASE_URL}/api/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            question: questionText,
            conversation_id: currentConversationId,
            stream: true
        })
        });

        if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
            if (line.startsWith('data: ')) {
            const data = line.slice(6);
            
            if (data === '[DONE]') {
                setStreamingStatus(null);
                setIsStreaming(false);
                setIsLoading(false);
                return;
            }

            try {
                const parsed = JSON.parse(data);
                
                if (parsed.error) {
                throw new Error(parsed.error);
                }

                // Only handle the final complete response
                if (parsed.status === 'complete') {
                setCurrentConversationId(parsed.conversation_id);
                
                const finalMessage = {
                    id: Date.now() + 1,
                    type: 'assistant',
                    content: parsed.answer,
                    visualizations: parsed.visualizations,
                    keyMetrics: parsed.key_metrics,
                    functionsExecuted: parsed.functions_executed,
                    executionSummary: parsed.execution_summary,
                    dataSources: parsed.data_sources,
                    timestamp: new Date(),
                    isStreaming: false,
                    hasContext: parsed.has_context || false  // Add this line
                };

                setConversations(prev => [...prev, finalMessage]);
                setStreamingStatus(null);
                setIsStreaming(false);
                setIsLoading(false);
                }
                // Ignore any other status updates

            } catch (parseError) {
                console.error('Error parsing stream data:', parseError);
            }
            }
        }
        }

    } catch (error) {
        console.error('Streaming query failed:', error);
        const errorMessage = {
        id: Date.now() + 1,
        type: 'error',
        content: `Failed to process query: ${error.message}. Please check that the FastAPI server is running on ${API_BASE_URL}`,
        timestamp: new Date()
        };
        setConversations(prev => [...prev, errorMessage]);
        setStreamingStatus(null);
        setIsStreaming(false);
        setIsLoading(false);
        setError(error.message);
    }
    };

  const handlePDFDownload = (conversationId) => {
    console.log('Starting PDF download for conversation:', conversationId);
  };

  const handlePDFComplete = (success, error) => {
    if (success) {
      console.log('PDF downloaded successfully');
    } else {
      console.error('PDF download failed:', error);
    }
  };

  const clearConversation = () => {
    setConversations([]);
    setCurrentConversationId(null);
    setError(null);
  };

  return (
    <div className="revenue-dashboard">
      {/* Header */}
      <header className="dashboard-header">
        <div className="header-content">
          <div className="header-top">
            <div className="header-left">
              <div className="logo">
                <BarChart3 size={24} />
              </div>
              <div className="header-text">
                <h1>Revenue Analytics</h1>
                <p>AI-Powered Executive Intelligence</p>
              </div>
            </div>
            <div className="header-right">
              <SystemStatusIndicator systemStatus={systemStatus} onRefresh={checkSystemHealth} />
              {conversations.length > 0 && (
                <button onClick={clearConversation} className="clear-btn" title="Clear conversation">
                  <RefreshCw size={16} />
                  <span>Clear</span>
                </button>
              )}
            </div>
          </div>
          
          {/* Tab Navigation */}
          <div className="tab-navigation">
            {[
              { id: 'chat', label: 'Analytics Chat', icon: Search },
              { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
              >
                <tab.icon size={16} />
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div className="error-banner">
          <AlertTriangle size={16} />
          <span>{error}</span>
          <button onClick={() => setError(null)} className="error-close">×</button>
        </div>
      )}

      {/* Main Content */}
      <main className="main-content">
        {activeTab === 'chat' && (
          <div className="chat-layout">
            {/* Quick Questions Sidebar */}
            <div className="sidebar">
              <h3>Quick Questions</h3>
              <div className="quick-questions">
                {quickQuestions.map((question, index) => (
                  <button
                    key={index}
                    onClick={() => handleStreamingQuery(question)}
                    className="quick-question"
                    disabled={isStreaming}
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>

            {/* Chat Interface */}
            <div className="chat-container">
              {/* Chat Messages */}
              <div className="chat-messages">
                {conversations.length === 0 && (
                  <div className="welcome-screen">
                    <Search size={48} className="welcome-icon" />
                    <h3>Welcome to Revenue Analytics</h3>
                    <p>Ask me anything about your sales data, opportunities, and accounts.</p>
                    <div className="capability-badges">
                      <span className="badge pipeline">
                        <TrendingUp size={12} />
                        Pipeline Analysis
                      </span>
                      <span className="badge revenue">
                        <DollarSign size={12} />
                        Revenue Insights
                      </span>
                      <span className="badge accounts">
                        <Users size={12} />
                        Account Intelligence
                      </span>
                    </div>
                  </div>
                )}

                {conversations.map((message) => (
                    <div key={message.id} className={`message-container ${message.type}`}>
                        <div className={`message ${message.type}`}>
                        {message.hasContext && message.type === 'assistant' && (
                            <ContextIndicator hasContext={true} />
                        )}
                        <div 
                            className="message-content"
                            dangerouslySetInnerHTML={{ __html: formatMessageContent(message.content) }}
                        />
                        
                        {(message.visualizations || message.keyMetrics) && (
                            <VisualizationsContainer 
                            visualizations={message.visualizations}
                            keyMetrics={message.keyMetrics}
                            dataSources={message.dataSources}
                            />
                        )}
                        
                        <div className="message-meta">
                            <span>{message.timestamp.toLocaleTimeString()}</span>
                            {message.executionSummary && (
                            <span> • {message.executionSummary}</span>
                            )}
                            {message.dataSources && message.dataSources.length > 0 && (
                            <DataSourcesCitation 
                                dataSources={message.dataSources} 
                                className="message-citation" 
                            />
                            )}
                            {message.type === 'assistant' && !message.isStreaming && currentConversationId && (
                            <PDFDownloadButton
                                conversationId={currentConversationId}
                                disabled={isStreaming}
                                onDownloadStart={() => handlePDFDownload(currentConversationId)}
                                onDownloadComplete={handlePDFComplete}
                            />
                            )}
                        </div>
                        </div>
                    </div>
                    ))}

                
                {isStreaming && streamingStatus && (
                  <div className="message-container assistant">
                    <div className="message assistant streaming">
                      <StreamingStatus 
                        status={streamingStatus.status}
                        message={streamingStatus.message}
                        iteration={streamingStatus.iteration}
                        currentFunction={streamingStatus.currentFunction}
                      />
                    </div>
                  </div>
                )}
                
                <div ref={chatEndRef} />
              </div>

              {/* Input Area */}
              <div className="input-area">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleStreamingQuery()}
                  placeholder="Ask me about your revenue, accounts, opportunities..."
                  className="query-input"
                  disabled={isStreaming}
                />
                <button
                  onClick={() => handleStreamingQuery()}
                  disabled={!query.trim() || isStreaming}
                  className="submit-button"
                >
                  {isStreaming ? <Loader2 size={16} className="spinner" /> : <Send size={16} />}
                  <span>{isStreaming ? 'Analyzing...' : 'Ask'}</span>
                </button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'dashboard' && (
          <div className="dashboard-placeholder">
            <BarChart3 size={64} />
            <h3>Dashboard Coming Soon</h3>
            <p>Interactive dashboards and widgets will be available soon.</p>
            <p className="sub-text">For now, use the Analytics Chat to explore your data with natural language queries.</p>
          </div>
        )}
      </main>

    </div>
  );
};

export default RevenueAnalyticsDashboard;