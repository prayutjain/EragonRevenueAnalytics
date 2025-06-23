#!/usr/bin/env python3
"""
Enhanced Revenue Analytics Backend - Comprehensive CRO Analytics System
Usage: python enhanced_analytics_server.py
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union, AsyncGenerator
import asyncio
import re
import base64
import io

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
import openai
from openai import OpenAI
import uvicorn

# PDF Gen
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.lib.colors import HexColor
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from collections import deque
import statistics

# Environment check
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Missing OPENAI_API_KEY environment variable")
    print("Set it with: export OPENAI_API_KEY='your-key-here'")
    exit(1)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Revenue Analytics API",
    description="Comprehensive AI-powered analytics system for enterprise CROs",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global data storage
opportunities_df = None
contacts_df = None

# Pydantic models
class StreamingQueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    max_iterations: Optional[int] = 3
    stream: bool = True

class PDFReportRequest(BaseModel):
    conversation_id: str
    title: Optional[str] = "Revenue Analytics Report"
    include_visualizations: bool = True
    include_insights: bool = True

class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    max_iterations: Optional[int] = 3

class SummarizedData(BaseModel):
    """Summarized and presentation-ready data"""
    title: str
    description: str
    chart_type: str  # 'bar', 'pie', 'line', 'table', 'metrics'
    data: List[Dict[str, Any]]
    key_insights: List[str]
    source_functions: List[str]
    data_sources: Optional[List[str]] = None

class QueryResponse(BaseModel):
    answer: str
    visualizations: Optional[List[SummarizedData]] = None
    key_metrics: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None
    functions_executed: Optional[List[str]] = None
    execution_summary: Optional[str] = None
    data_sources: Optional[List[str]] = None
    streaming_complete: Optional[bool] = None

class HealthResponse(BaseModel):
    status: str
    data_loaded: bool
    timestamp: str
    records_count: Dict[str, int]

# Utility functions
def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj
    
def parse_product_service(opportunity_name: str) -> Dict[str, str]:
    """Parse opportunity name to extract product category and service type"""
    if pd.isna(opportunity_name):
        return {"category": "Unknown", "service": "Unknown", "action_type": "Unknown"}
    
    parts = opportunity_name.split(':')
    category = parts[0].strip() if len(parts) > 0 else "Unknown"
    service_part = parts[1].strip() if len(parts) > 1 else "Unknown"
    
    # Extract action type (Implementation, Adoption Initiative, etc.)
    action_patterns = [
        "Adoption Initiative", "Expansion Project", "Implementation", 
        "Upgrade Proposal", "Migration", "Enhancement"
    ]
    
    action_type = "Unknown"
    for pattern in action_patterns:
        if pattern in service_part:
            action_type = pattern
            service = service_part.replace(pattern, "").strip()
            break
    else:
        service = service_part
    
    return {
        "category": category,
        "service": service,
        "action_type": action_type
    }

def calculate_weighted_value(amount: float, probability: float) -> float:
    """Calculate weighted opportunity value"""
    return amount * (probability / 100.0)

def days_in_stage(created_date: str, close_date: str = None) -> int:
    """Calculate days an opportunity has been in current stage"""
    try:
        created = pd.to_datetime(created_date)
        end_date = datetime.now()
        return (end_date - created).days
    except:
        return 0

# Store conversation history for PDF generation
conversation_history = {}
conversation_contexts = {}

# PDF Generation Functions
def create_chart_image(visualization: SummarizedData) -> Optional[str]:
    """Create chart image for PDF report"""
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = visualization.data
        chart_type = visualization.chart_type
        
        if chart_type == 'bar' and data:
            names = [item.get('name', '') for item in data]
            values = [item.get('value', 0) for item in data]
            
            bars = ax.bar(names, values, color=['#8B5CF6', '#06B6D4', '#10B981', '#F59E0B', '#EF4444'])
            ax.set_title(visualization.title, fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel('Value')
            
            # Format y-axis
            if max(values) > 1000000:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000000:.1f}M'))
            elif max(values) > 1000:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
            
            # Rotate x-axis labels if needed
            plt.xticks(rotation=45, ha='right')
            
        elif chart_type == 'pie' and data:
            names = [item.get('name', '') for item in data]
            values = [item.get('value', 0) for item in data]
            
            colors_list = ['#8B5CF6', '#06B6D4', '#10B981', '#F59E0B', '#EF4444', '#8B5A2B', '#6366F1']
            ax.pie(values, labels=names, autopct='%1.1f%%', colors=colors_list[:len(values)])
            ax.set_title(visualization.title, fontsize=14, fontweight='bold', pad=20)
            
        elif chart_type == 'line' and data:
            if len(data) > 0:
                periods = [item.get('period', '') for item in data]
                # Get numeric columns
                numeric_keys = [k for k in data[0].keys() if k != 'period' and isinstance(data[0].get(k), (int, float))]
                
                for key in numeric_keys:
                    values = [item.get(key, 0) for item in data]
                    ax.plot(periods, values, marker='o', linewidth=2, label=key)
                
                ax.set_title(visualization.title, fontsize=14, fontweight='bold', pad=20)
                ax.set_ylabel('Value')
                ax.legend()
                plt.xticks(rotation=45, ha='right')
        
        else:
            # For metrics or unsupported types, create a simple text display
            ax.text(0.5, 0.5, visualization.title, transform=ax.transAxes, 
                   fontsize=16, ha='center', va='center', fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return image_base64
        
    except Exception as e:
        print(f"Error creating chart image: {e}")
        plt.close('all')
        return None

# Updated PDF Generation Functions - Replace the existing functions in your code

def clean_text_for_pdf(text: str) -> str:
    """Clean text for PDF without HTML tags - for table cells and general text"""
    if not text:
        return ""
    
    text = str(text)
    
    # Remove any HTML tags completely
    text = re.sub(r'<[^>]+>', '', text)
    
    # Fix common HTML entities
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    
    # Remove problematic characters that might break PDF
    text = re.sub(r'[^\x00-\x7F\u00A0-\u024F\u1E00-\u1EFF]', '', text)
    
    # Clean up multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def safe_format_value(value, header=""):
    """Safely format values for PDF table display"""
    if value is None or value == '' or pd.isna(value):
        return ''
    
    # Convert to string first for safety
    str_value = str(value).strip()
    
    # Handle empty strings
    if not str_value:
        return ''
    
    # Try to detect and format numeric values
    try:
        # Clean numeric strings
        cleaned_value = str_value.replace(',', '').replace('$', '').replace('%', '').replace(' ', '')
        
        # Check if it's a valid number
        if cleaned_value.replace('.', '').replace('-', '').isdigit():
            numeric_value = float(cleaned_value)
            
            # Format based on column name and value size
            header_lower = header.lower()
            
            if 'amount' in header_lower or 'value' in header_lower or 'revenue' in header_lower or 'cost' in header_lower or 'price' in header_lower:
                # Currency formatting
                if abs(numeric_value) >= 1000000:
                    return f"${numeric_value/1000000:.1f}M"
                elif abs(numeric_value) >= 1000:
                    return f"${numeric_value/1000:.0f}K"
                else:
                    return f"${numeric_value:,.0f}"
            
            elif 'percent' in header_lower or 'rate' in header_lower or 'probability' in header_lower:
                # Percentage formatting
                return f"{numeric_value:.1f}%"
            
            elif 'count' in header_lower or 'deals' in header_lower or 'days' in header_lower:
                # Integer formatting
                return f"{int(numeric_value):,}"
            
            else:
                # General numeric formatting
                if numeric_value >= 1000000:
                    return f"{numeric_value/1000000:.1f}M"
                elif numeric_value >= 1000:
                    return f"{numeric_value/1000:.0f}K"
                elif numeric_value == int(numeric_value):
                    return f"{int(numeric_value):,}"
                else:
                    return f"{numeric_value:.2f}"
    
    except (ValueError, TypeError, AttributeError):
        pass
    
    # If not numeric or formatting failed, return cleaned string
    cleaned_str = clean_text_for_pdf(str_value)
    
    # Truncate very long strings for table display
    if len(cleaned_str) > 40:
        cleaned_str = cleaned_str[:37] + "..."
    
    return cleaned_str

def create_formatted_table_data(data_list, max_rows=20):
    """Create properly formatted table data for PDF"""
    if not data_list or len(data_list) == 0:
        return [], []
    
    # Limit the number of rows for better PDF layout
    display_data = data_list[:max_rows]
    
    # Get all unique keys from the data
    all_keys = set()
    for item in display_data:
        if isinstance(item, dict):
            all_keys.update(item.keys())
    
    # Sort keys for consistent column order, prioritizing important ones first
    priority_keys = ['name', 'account_name', 'opportunity_name', 'stage', 'amount', 'value', 'total_value', 'count', 'probability']
    
    def sort_key_priority(key):
        key_lower = str(key).lower()
        for i, priority in enumerate(priority_keys):
            if priority in key_lower:
                return (0, i)  # High priority
        return (1, key_lower)  # Lower priority, alphabetical
    
    sorted_keys = sorted(all_keys, key=sort_key_priority)
    
    # Limit number of columns to fit page width (max 6-7 columns)
    if len(sorted_keys) > 6:
        sorted_keys = sorted_keys[:6]
    
    # Create header row
    headers = []
    for key in sorted_keys:
        clean_header = clean_text_for_pdf(str(key).replace('_', ' ').title())
        # Truncate long headers
        if len(clean_header) > 15:
            clean_header = clean_header[:12] + "..."
        headers.append(clean_header)
    
    # Create data rows
    table_rows = [headers]  # Start with header row
    
    for item in display_data:
        row = []
        for key in sorted_keys:
            value = item.get(key, '') if isinstance(item, dict) else ''
            formatted_value = safe_format_value(value, key)
            row.append(formatted_value)
        table_rows.append(row)
    
    return table_rows, sorted_keys

def generate_pdf_report(conversation_data: Dict, title: str = "Revenue Analytics Report") -> bytes:
    """Generate PDF report with properly formatted tables"""
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=A4, 
        rightMargin=40, 
        leftMargin=40, 
        topMargin=40, 
        bottomMargin=40,
        title=title
    )
    
    # Enhanced styles
    styles = getSampleStyleSheet()
    
    # Custom styles for better typography
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        spaceBefore=20,
        textColor=HexColor('#1a202c'),
        alignment=1,  # Center alignment
        fontName='Helvetica-Bold'
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=15,
        spaceBefore=20,
        textColor=HexColor('#2d3748'),
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10,
        spaceBefore=15,
        textColor=HexColor('#4a5568'),
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=10,
        spaceBefore=5,
        leading=14,
        textColor=HexColor('#2d3748'),
        fontName='Helvetica'
    )
    
    insight_style = ParagraphStyle(
        'InsightStyle',
        parent=styles['Normal'],
        fontSize=9,
        spaceAfter=6,
        spaceBefore=3,
        leading=12,
        leftIndent=15,
        rightIndent=10,
        textColor=HexColor('#5a67d8'),
        fontName='Helvetica-Oblique',
        backColor=HexColor('#f7fafc')
    )
    
    caption_style = ParagraphStyle(
        'CaptionStyle',
        parent=styles['Normal'],
        fontSize=8,
        spaceAfter=3,
        spaceBefore=2,
        leading=10,
        textColor=HexColor('#718096'),
        fontName='Helvetica-Oblique',
        alignment=1  # Center alignment
    )
    
    # Build story
    story = []
    
    # Title page
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading1_style))
    
    if 'query' in conversation_data:
        query_text = clean_text_for_pdf(conversation_data['query'])
        story.append(Paragraph(f"<b>Analysis Request:</b> {query_text}", body_style))
        story.append(Spacer(1, 12))
    
    if 'answer' in conversation_data:
        # Process the answer with proper formatting
        answer_sections = process_markdown_content(conversation_data['answer'])
        
        for section in answer_sections:
            if section['type'] == 'heading1':
                story.append(Paragraph(section['content'], heading1_style))
            elif section['type'] == 'heading2':
                story.append(Paragraph(section['content'], heading2_style))
            elif section['type'] == 'paragraph':
                story.append(Paragraph(section['content'], body_style))
            elif section['type'] == 'bullet_list':
                for item in section['items']:
                    story.append(Paragraph(f"• {item}", body_style))
    
    story.append(Spacer(1, 20))
    
    # Key Metrics Section
    if 'key_metrics' in conversation_data and conversation_data['key_metrics']:
        story.append(Paragraph("Key Performance Indicators", heading1_style))
        
        # Create metrics table
        metrics_data = []
        
        for key, metric in conversation_data['key_metrics'].items():
            if isinstance(metric, dict):
                label = metric.get('label', key.replace('_', ' ').title())
                value = metric.get('value', 0)
                format_type = metric.get('format', 'number')
            else:
                label = key.replace('_', ' ').title()
                value = metric
                format_type = 'number'
            
            # Format value
            if format_type == 'currency':
                formatted_value = f"${value:,.0f}"
            elif format_type == 'percentage':
                formatted_value = f"{value:.1f}%"
            else:
                formatted_value = f"{value:,.0f}"
            
            metrics_data.append([label, formatted_value])
        
        # Create metrics table
        if metrics_data:
            available_width = doc.width
            metrics_table = Table(metrics_data, colWidths=[available_width * 0.7, available_width * 0.3])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), HexColor('#f8fafc')),
                ('TEXTCOLOR', (0, 0), (-1, -1), HexColor('#2d3748')),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#e2e8f0')),
            ]))
            story.append(metrics_table)
            story.append(Spacer(1, 20))
    
    # Visualizations Section
    if 'visualizations' in conversation_data and conversation_data['visualizations']:
        story.append(Paragraph("Detailed Analysis &amp; Visualizations", heading1_style))
        
        for i, viz in enumerate(conversation_data['visualizations']):
            if i > 0:
                story.append(PageBreak())  # New page for each major visualization
            
            # Visualization title
            viz_title = clean_text_for_pdf(viz.get('title', f'Visualization {i+1}'))
            story.append(Paragraph(viz_title, heading2_style))
            
            # Description
            if viz.get('description'):
                description = clean_text_for_pdf(viz['description'])
                story.append(Paragraph(description, body_style))
                story.append(Spacer(1, 10))
            
            # Data sources citation
            if viz.get('data_sources'):
                sources_text = f"Data Sources: {', '.join(viz['data_sources'])}"
                story.append(Paragraph(sources_text, caption_style))
                story.append(Spacer(1, 10))
            
            # Create and add chart image (for non-table visualizations)
            if viz.get('chart_type') != 'table':
                try:
                    chart_image_base64 = create_chart_image(SummarizedData(**viz))
                    if chart_image_base64:
                        image_data = base64.b64decode(chart_image_base64)
                        image_buffer = io.BytesIO(image_data)
                        
                        # Responsive image sizing
                        max_width = doc.width - 20
                        max_height = 250
                        img = Image(image_buffer, width=max_width, height=max_height)
                        img.hAlign = 'CENTER'
                        story.append(img)
                        story.append(Spacer(1, 15))
                except Exception as e:
                    print(f"Error adding chart to PDF: {e}")
                    story.append(Paragraph("Chart visualization unavailable", caption_style))
            
            # Enhanced data table for table visualizations
            if viz.get('chart_type') == 'table' and viz.get('data'):
                table_data = viz['data']
                
                if table_data and len(table_data) > 0:
                    # Create properly formatted table
                    table_rows, column_keys = create_formatted_table_data(table_data, max_rows=15)
                    
                    if table_rows and len(table_rows) > 1:  # Has header + data
                        # Calculate column widths
                        available_width = doc.width - 20
                        num_cols = len(table_rows[0])
                        base_width = available_width / num_cols
                        
                        # Adjust column widths based on content
                        col_widths = []
                        for col_idx in range(num_cols):
                            # Check header length
                            header_len = len(str(table_rows[0][col_idx]))
                            
                            # Check a few data rows for content length
                            max_data_len = 0
                            for row_idx in range(1, min(len(table_rows), 6)):  # Check first 5 data rows
                                if col_idx < len(table_rows[row_idx]):
                                    data_len = len(str(table_rows[row_idx][col_idx]))
                                    max_data_len = max(max_data_len, data_len)
                            
                            # Calculate width based on content
                            content_width = max(header_len, max_data_len) * 6  # Approximate character width
                            optimal_width = max(base_width * 0.6, min(content_width, base_width * 1.5))
                            col_widths.append(optimal_width)
                        
                        # Normalize to fit available width
                        total_width = sum(col_widths)
                        if total_width > available_width:
                            scale_factor = available_width / total_width
                            col_widths = [w * scale_factor for w in col_widths]
                        
                        # Create the table
                        data_table = Table(table_rows, colWidths=col_widths, repeatRows=1)
                        data_table.setStyle(TableStyle([
                            # Header styling
                            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#4a5568')),
                            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 8),
                            
                            # Body styling
                            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#ffffff')),
                            ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#2d3748')),
                            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                            ('FONTSIZE', (0, 1), (-1, -1), 7),
                            
                            # Alternating row colors
                            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#f8fafc'), HexColor('#ffffff')]),
                            
                            # Borders and alignment
                            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#e2e8f0')),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                            ('TOPPADDING', (0, 0), (-1, -1), 4),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                            ('LEFTPADDING', (0, 0), (-1, -1), 5),
                            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                        ]))
                        
                        story.append(data_table)
                        
                        # Add note if data was truncated
                        if len(table_data) > 15:
                            story.append(Spacer(1, 5))
                            story.append(Paragraph(f"Note: Showing first 15 of {len(table_data)} total records", 
                                                caption_style))
                
                story.append(Spacer(1, 15))
            
            # Key insights
            if viz.get('key_insights'):
                story.append(Paragraph("Key Insights:", heading2_style))
                for insight in viz['key_insights']:
                    clean_insight = clean_text_for_pdf(insight)
                    story.append(Paragraph(f"• {clean_insight}", insight_style))
                story.append(Spacer(1, 15))
    
    # Footer section
    story.append(Spacer(1, 30))
    story.append(Paragraph("Report Details", heading2_style))
    
    footer_info = [
        ['Generated:', datetime.now().strftime('%B %d, %Y at %H:%M:%S')],
        ['System:', 'Revenue Analytics AI Platform v2.1'],
    ]
    
    if 'data_sources' in conversation_data and conversation_data['data_sources']:
        footer_info.append(['Data Sources:', ', '.join(conversation_data['data_sources'])])
    
    if 'functions_executed' in conversation_data and conversation_data['functions_executed']:
        footer_info.append(['Analysis Functions:', ', '.join(conversation_data['functions_executed'])])
    
    footer_table = Table(footer_info, colWidths=[2*inch, 4*inch])
    footer_table.setStyle(TableStyle([
        ('TEXTCOLOR', (0, 0), (-1, -1), HexColor('#718096')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#e2e8f0')),
    ]))
    story.append(footer_table)
    
    # Build PDF with error handling
    try:
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        print(f"Error building PDF: {e}")
        # Create a minimal fallback PDF
        buffer = io.BytesIO()
        fallback_doc = SimpleDocTemplate(buffer, pagesize=A4)
        fallback_story = [
            Paragraph(title, title_style),
            Spacer(1, 20),
            Paragraph("Report Generation Error", heading1_style),
            Paragraph("The report encountered formatting issues during generation. Please try again or contact support.", body_style),
            Spacer(1, 20),
            Paragraph(f"Error details: {str(e)}", caption_style)
        ]
        fallback_doc.build(fallback_story)
        buffer.seek(0)
        return buffer.getvalue()

def process_markdown_content(content: str) -> List[Dict[str, any]]:
    """Process markdown content into structured sections for better PDF formatting"""
    if not content:
        return []
    
    sections = []
    lines = content.split('\n')
    current_paragraph = []
    current_list = []
    list_type = None
    
    for line in lines:
        line = line.strip()
        
        if not line:
            # Empty line - end current paragraph or list
            if current_paragraph:
                sections.append({
                    'type': 'paragraph',
                    'content': clean_text_for_pdf(' '.join(current_paragraph))
                })
                current_paragraph = []
            
            if current_list:
                sections.append({
                    'type': list_type,
                    'items': [clean_text_for_pdf(item) for item in current_list]
                })
                current_list = []
                list_type = None
            continue
        
        # Check for headings
        if line.startswith('### '):
            # End current content
            if current_paragraph:
                sections.append({'type': 'paragraph', 'content': clean_text_for_pdf(' '.join(current_paragraph))})
                current_paragraph = []
            if current_list:
                sections.append({'type': list_type, 'items': [clean_text_for_pdf(item) for item in current_list]})
                current_list = []
                list_type = None
            
            sections.append({'type': 'heading3', 'content': clean_text_for_pdf(line[4:])})
            
        elif line.startswith('## '):
            # End current content
            if current_paragraph:
                sections.append({'type': 'paragraph', 'content': clean_text_for_pdf(' '.join(current_paragraph))})
                current_paragraph = []
            if current_list:
                sections.append({'type': list_type, 'items': [clean_text_for_pdf(item) for item in current_list]})
                current_list = []
                list_type = None
            
            sections.append({'type': 'heading2', 'content': clean_text_for_pdf(line[3:])})
            
        elif line.startswith('# '):
            # End current content
            if current_paragraph:
                sections.append({'type': 'paragraph', 'content': clean_text_for_pdf(' '.join(current_paragraph))})
                current_paragraph = []
            if current_list:
                sections.append({'type': list_type, 'items': [clean_text_for_pdf(item) for item in current_list]})
                current_list = []
                list_type = None
            
            sections.append({'type': 'heading1', 'content': clean_text_for_pdf(line[2:])})
            
        elif line.startswith('- ') or line.startswith('* '):
            # Bullet list item
            if current_paragraph:
                sections.append({'type': 'paragraph', 'content': clean_text_for_pdf(' '.join(current_paragraph))})
                current_paragraph = []
            
            if list_type != 'bullet_list':
                if current_list:
                    sections.append({'type': list_type, 'items': [clean_text_for_pdf(item) for item in current_list]})
                current_list = []
                list_type = 'bullet_list'
            
            current_list.append(line[2:])
            
        elif re.match(r'^\d+\.\s', line):
            # Numbered list item
            if current_paragraph:
                sections.append({'type': 'paragraph', 'content': clean_text_for_pdf(' '.join(current_paragraph))})
                current_paragraph = []
            
            if list_type != 'numbered_list':
                if current_list:
                    sections.append({'type': list_type, 'items': [clean_text_for_pdf(item) for item in current_list]})
                current_list = []
                list_type = 'numbered_list'
            
            # Remove the number and dot
            item_text = re.sub(r'^\d+\.\s*', '', line)
            current_list.append(item_text)
            
        else:
            # Regular paragraph text
            if current_list:
                sections.append({'type': list_type, 'items': [clean_text_for_pdf(item) for item in current_list]})
                current_list = []
                list_type = None
            
            current_paragraph.append(line)
    
    # Handle remaining content
    if current_paragraph:
        sections.append({'type': 'paragraph', 'content': clean_text_for_pdf(' '.join(current_paragraph))})
    
    if current_list:
        sections.append({'type': list_type, 'items': [clean_text_for_pdf(item) for item in current_list]})
    
    return sections


# Data loading functions
def load_csv_data():
    """Load and validate CSV data with enhanced processing"""
    global opportunities_df, contacts_df
    
    required_files = ['opportunities.csv', 'account_and_contact.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing required CSV files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure these files are in the same directory as this script.")
        exit(1)
    
    try:
        print("📈 Loading opportunities data...")
        opportunities_df = pd.read_csv('opportunities.csv')
        
        print("👥 Loading contacts data...")
        contacts_df = pd.read_csv('account_and_contact.csv')
        
        # Enhanced data cleaning and preparation
        opportunities_df['Amount'] = pd.to_numeric(opportunities_df['Amount'], errors='coerce').fillna(0)
        opportunities_df['Probability (%)'] = pd.to_numeric(opportunities_df['Probability (%)'], errors='coerce').fillna(0)
        opportunities_df['Close Date'] = pd.to_datetime(opportunities_df['Close Date'], errors='coerce')
        opportunities_df['Created Date'] = pd.to_datetime(opportunities_df['Created Date'], errors='coerce')
        
        # Add calculated fields
        opportunities_df['Weighted Value'] = opportunities_df.apply(
            lambda row: calculate_weighted_value(row['Amount'], row['Probability (%)']), axis=1
        )
        
        # Parse product/service information
        product_info = opportunities_df['Opportunity Name'].apply(parse_product_service)
        opportunities_df['Product Category'] = [info['category'] for info in product_info]
        opportunities_df['Service Type'] = [info['service'] for info in product_info]
        opportunities_df['Action Type'] = [info['action_type'] for info in product_info]
        
        # Calculate days in stage
        opportunities_df['Days in Stage'] = opportunities_df.apply(
            lambda row: days_in_stage(row['Created Date'], row['Close Date']), axis=1
        )
        
        # Add fiscal quarter
        opportunities_df['Close Quarter'] = opportunities_df['Close Date'].dt.to_period('Q').astype(str)
        
        print(f"✅ Loaded {len(opportunities_df)} opportunities and {len(contacts_df)} contacts")
        print(f"📊 Added calculated fields: Weighted Value, Product Category, Service Type, Days in Stage")
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        exit(1)

# PERFORMANCE & FORECASTING FUNCTIONS

def get_top_opportunities_by_value(
    limit: int = 10,
    stage_filter: Optional[str] = None,
    include_probability: bool = True
) -> List[Dict]:
    """Get top opportunities by deal value with stage and probability info"""
    
    df = opportunities_df.copy()
    
    if stage_filter:
        df = df[df['Stage'] == stage_filter]

    # By default ignore Closed deals
    df = df[~(df['Stage'].isin(["Closed Won", "Closed Lost"]))]
    
    # Sort by amount descending
    top_opps = df.nlargest(limit, 'Amount')
    
    return [
        {
            "opportunity_name": row['Opportunity Name'],
            "account_name": row['Account Name'],
            "amount": float(row['Amount']),
            "stage": row['Stage'],
            "probability": int(row['Probability (%)']),
            "weighted_value": round(float(row['Weighted Value']), 2),
            "close_date": row['Close Date'].strftime('%Y-%m-%d') if pd.notna(row['Close Date']) else None,
            "product_category": row['Product Category'],
            "service_type": row['Service Type']
        }
        for _, row in top_opps.iterrows()
    ]

def get_high_probability_revenue_forecast(
    min_probability: int = 70,
    fiscal_period: Optional[str] = None,
    exclude_closed_won: bool = True
) -> List[Dict[str, Any]]:
    """Calculate expected revenue from high-probability deals"""
    
    df = opportunities_df.copy()
    
    # Filter high probability deals
    df = df[df['Probability (%)'] >= min_probability]
    
    # Exclude closed lost - this is anyway given (Lost is generally prob <15)
    df = df[df['Stage'] != 'Closed Lost']

    # Exclude closed won if specified
    if exclude_closed_won:
        df = df[df['Stage'] != 'Closed Won']
    
    if fiscal_period:
        df = df[df['Fiscal Period'] == fiscal_period]
    
    total_amount = df['Amount'].sum()
    weighted_revenue = df['Weighted Value'].sum()
    deal_count = len(df)
    avg_probability = df['Probability (%)'].mean() if deal_count > 0 else 0
    
    # Breakdown by stage
    stage_breakdown = df.groupby('Stage').agg({
        'Amount': 'sum',
        'Weighted Value': 'sum',
        'Opportunity Name': 'count'
    }).round(2)
    
    # Return as a flat list for visualization
    result = []
    
    # Add summary row
    result.append({
        "stage": "TOTAL FORECAST",
        "deal_count": deal_count,
        "total_value": float(total_amount),
        "expected_value": float(weighted_revenue),
        "average_probability": float(avg_probability),
        "type": "summary"
    })
    
    # Add stage breakdown
    for stage, row in stage_breakdown.iterrows():
        result.append({
            "stage": stage,
            "deal_count": int(row['Opportunity Name']),
            "total_value": float(row['Amount']),
            "expected_value": float(row['Weighted Value']),
            "average_probability": float(df[df['Stage'] == stage]['Probability (%)'].mean()),
            "type": "stage"
        })
    
    return result

def analyze_product_win_rates() -> List[Dict]:
    """Analyze win rates by product/service categories"""
    
    df = opportunities_df.copy()
    
    # Only include deals that have reached a conclusion
    concluded_deals = df[df['Stage'].isin(['Closed Won', 'Closed Lost'])]
    
    # Group by product category and service
    product_analysis = concluded_deals.groupby(['Product Category', 'Service Type', 'Stage']).size().unstack(fill_value=0)
    
    results = []
    for (category, service), row in product_analysis.iterrows():
        won = row.get('Closed Won', 0)
        lost = row.get('Closed Lost', 0)
        total = won + lost
        
        if total > 0:
            win_rate = (won / total) * 100
            results.append({
                "product_category": category,
                "service_type": service,
                "deals_won": int(won),
                "deals_lost": int(lost),
                "total_deals": int(total),
                "win_rate_percent": round(win_rate, 1)
            })
    
    return sorted(results, key=lambda x: x['win_rate_percent'], reverse=True)

def analyze_win_rates_by_type() -> List[Dict]:
    """Analyze win rates by opportunity type (New Business, Expansion, Renewal)"""
    
    df = opportunities_df.copy()
    concluded_deals = df[df['Stage'].isin(['Closed Won', 'Closed Lost'])]
    
    type_analysis = concluded_deals.groupby(['Type', 'Stage']).size().unstack(fill_value=0)
    
    results = []
    for opp_type, row in type_analysis.iterrows():
        if pd.isna(opp_type):
            continue
            
        won = row.get('Closed Won', 0)
        lost = row.get('Closed Lost', 0)
        total = won + lost
        
        if total > 0:
            win_rate = (won / total) * 100
            avg_deal_size = concluded_deals[concluded_deals['Type'] == opp_type]['Amount'].mean()
            
            results.append({
                "opportunity_type": opp_type,
                "deals_won": int(won),
                "deals_lost": int(lost),
                "total_deals": int(total),
                "win_rate_percent": round(win_rate, 1),
                "avg_deal_size": round(avg_deal_size, 2)
            })
    
    return sorted(results, key=lambda x: x['win_rate_percent'], reverse=True)

# Deprecate, only 1 fiscal period given - Q1 2025
def analyze_fiscal_period_distribution() -> List[Dict]:
    """Analyze deal distribution across fiscal periods"""
    
    df = opportunities_df.copy()
    
    # Exclude closed deals for pipeline analysis
    pipeline_deals = df[~df['Stage'].isin(['Closed Won', 'Closed Lost'])]
    
    period_analysis = pipeline_deals.groupby('Fiscal Period').agg({
        'Amount': ['sum', 'mean', 'count'],
        'Weighted Value': 'sum',
        'Probability (%)': 'mean'
    }).round(2)
    
    period_analysis.columns = ['total_value', 'avg_deal_size', 'deal_count', 'weighted_value', 'avg_probability']
    
    return [
        {
            "fiscal_period": period,
            "deal_count": int(row['deal_count']),
            "total_pipeline_value": float(row['total_value']),
            "expected_revenue": float(row['weighted_value']),
            "avg_deal_size": float(row['avg_deal_size']),
            "avg_probability": round(float(row['avg_probability']), 1)
        }
        for period, row in period_analysis.iterrows()
    ]

# ACCOUNT & OPPORTUNITY INSIGHTS FUNCTIONS

def get_top_accounts_by_total_value(
    limit: int = 5,
    include_closed_won: bool = True,
    include_closed_lost: bool = False
) -> List[Dict]:
    """Get top accounts by total opportunity value (won + open)"""
    
    df = opportunities_df.copy()
    
    if not include_closed_won:
        df = df[~(df['Stage'] == 'Closed Won')]

    if not include_closed_lost:
        df = df[~(df['Stage'] == 'Closed Lost')]
    
    account_metrics = df.groupby('Account Name').agg({
        'Amount': ['sum', 'mean', 'count'],
        'Weighted Value': 'sum',
        'Stage': lambda x: x.value_counts().to_dict()
    }).round(2)
    
    account_metrics.columns = ['total_value', 'avg_deal_size', 'deal_count', 'weighted_value', 'stage_distribution']
    
    top_accounts = account_metrics.nlargest(limit, 'total_value')
    
    return [
        {
            "account_name": account,
            "total_opportunity_value": float(row['total_value']),
            "expected_value": float(row['weighted_value']),
            "deal_count": int(row['deal_count']),
            "avg_deal_size": float(row['avg_deal_size']),
            "stage_distribution": row['stage_distribution']
        }
        for account, row in top_accounts.iterrows()
    ]

def get_expansion_opportunities() -> List[Dict]:
    """Get accounts with open Expansion opportunities and their stages"""
    
    df = opportunities_df.copy()
    
    # Filter for expansion opportunities that are not closed
    expansion_opps = df[
        (df['Type'] == 'Expansion') & 
        (~df['Stage'].isin(['Closed Won', 'Closed Lost']))
    ]
    
    return [
        {
            "account_name": row['Account Name'],
            "opportunity_name": row['Opportunity Name'],
            "stage": row['Stage'],
            "amount": float(row['Amount']),
            "probability": int(row['Probability (%)']),
            "next_step": row['Next Step'],
            "product_category": row['Product Category'],
            "days_in_stage": int(row['Days in Stage'])
        }
        for _, row in expansion_opps.iterrows()
    ]

def analyze_stalled_high_stage_deals() -> List[Dict]:
    """Find accounts that reach Negotiate/Propose but end up as Closed Lost"""
    
    df = opportunities_df.copy()
    
    # Get accounts with deals that were in late stages but lost
    late_stage_lost = df[
        (df['Stage'] == 'Closed Lost') & 
        (df['Next Step'].str.contains('Negotiate|Propose|Review the proposal|Finalize contract', case=False, na=False))
    ]
    
    # Group by account to see patterns
    account_patterns = late_stage_lost.groupby('Account Name').agg({
        'Opportunity Name': 'count',
        'Amount': 'sum',
        'Next Step': lambda x: list(x),
        'Product Category': lambda x: list(set(x))
    })
    
    return [
        {
            "account_name": account,
            "lost_deals_count": int(row['Opportunity Name']),
            "lost_value": float(row['Amount']),
            "common_next_steps": row['Next Step'],
            "product_categories": row['Product Category']
        }
        for account, row in account_patterns.iterrows()
    ]

def analyze_overdue_next_steps(days_threshold: int = 30) -> List[Dict]:
    """Find opportunities with overdue or unclear next steps"""
    
    df = opportunities_df.copy()
    
    # Filter for open opportunities
    open_opps = df[~df['Stage'].isin(['Closed Won', 'Closed Lost'])]
    
    # Find deals that have been in stage too long or have vague next steps
    overdue_deals = open_opps[
        (open_opps['Days in Stage'] > days_threshold) |
        (open_opps['Next Step'].str.contains('unclear|TBD|pending|follow up', case=False, na=False)) |
        (open_opps['Next Step'].isna())
    ]
    
    return [
        {
            "account_name": row['Account Name'],
            "opportunity_name": row['Opportunity Name'],
            "stage": row['Stage'],
            "amount": float(row['Amount']),
            "days_in_stage": int(row['Days in Stage']),
            "next_step": row['Next Step'] if pd.notna(row['Next Step']) else "No next step defined",
            "close_date": row['Close Date'].strftime('%Y-%m-%d') if pd.notna(row['Close Date']) else None,
            "urgency_score": int(row['Days in Stage']) + (0 if pd.notna(row['Next Step']) else 30)
        }
        for _, row in overdue_deals.iterrows()
    ]

def analyze_recent_losses(days_back: int = 30) -> List[Dict]:
    """Analyze recently lost deals and their next steps"""
    
    df = opportunities_df.copy()
    
    # Filter for recent losses
    cutoff_date = datetime.now() - timedelta(days=days_back)
    recent_losses = df[
        (df['Stage'] == 'Closed Lost') & 
        (df['Close Date'] >= cutoff_date)
    ]
    
    return [
        {
            "account_name": row['Account Name'],
            "opportunity_name": row['Opportunity Name'],
            "amount": float(row['Amount']),
            "close_date": row['Close Date'].strftime('%Y-%m-%d') if pd.notna(row['Close Date']) else None,
            "next_step": row['Next Step'],
            "product_category": row['Product Category'],
            "service_type": row['Service Type'],
            "primary_contact": row['Primary Contact'],
            "contact_title": row['Contact: Title']
        }
        for _, row in recent_losses.iterrows()
    ]

# CONTACT & ROLE INFLUENCE FUNCTIONS

def analyze_winning_contact_titles() -> List[Dict]:
    """Analyze which titles appear most in Closed Won deals"""
    
    df = opportunities_df.copy()
    won_deals = df[df['Stage'] == 'Closed Won']
    
    # Analyze both Primary Contact titles and Contact titles
    contact_analysis = won_deals['Contact: Title'].value_counts()
    
    total_won = len(won_deals)
    
    return [
        {
            "title": title,
            "won_deals_count": int(count),
            "percentage_of_wins": round((count / total_won) * 100, 1),
            "avg_deal_size": round(won_deals[won_deals['Contact: Title'] == title]['Amount'].mean(), 2)
        }
        for title, count in contact_analysis.head(15).items()
        if pd.notna(title)
    ]

def analyze_c_level_engagement() -> Dict[str, Any]:
    """Analyze C-level engagement in late-stage opportunities"""
    
    df = opportunities_df.copy()
    
    # Define C-level titles
    c_level_patterns = ['CEO', 'CTO', 'CFO', 'CRO', 'CMO', 'CPO', 'Chief']
    
    # Late stage opportunities
    late_stage = df[df['Stage'].isin(['Negotiate', 'Propose'])]
    
    # Check for C-level engagement
    late_stage['has_c_level'] = late_stage['Contact: Title'].str.contains('|'.join(c_level_patterns), case=False, na=False)
    
    c_level_stats = late_stage.groupby('has_c_level').agg({
        'Opportunity Name': 'count',
        'Amount': ['sum', 'mean'],
        'Probability (%)': 'mean'
    }).round(2)
    
    return {
        "late_stage_deals_with_c_level": int(c_level_stats.loc[True, ('Opportunity Name')] if True in c_level_stats.index else 0),
        "late_stage_deals_without_c_level": int(c_level_stats.loc[False, ('Opportunity Name')] if False in c_level_stats.index else 0),
        "c_level_avg_deal_size": float(c_level_stats.loc[True, ('Amount', 'mean')] if True in c_level_stats.index else 0),
        "non_c_level_avg_deal_size": float(c_level_stats.loc[False, ('Amount', 'mean')] if False in c_level_stats.index else 0),
        "c_level_avg_probability": float(c_level_stats.loc[True, ('Probability (%)', 'mean')] if True in c_level_stats.index else 0)
    }

def analyze_successful_actions() -> List[Dict]:
    """Analyze actions associated with successful CRO deals"""
    
    df = opportunities_df.copy()
    
    # Analyze by outcome
    won_deals = df[df['Stage'] == 'Closed Won']
    
    if len(won_deals) == 0:
        return []
    
    # Analyze common next steps in winning CRO deals
    next_step_analysis = won_deals['Next Step'].value_counts()
    
    return [
        {
            "action": action,
            "frequency": int(count),
            "avg_deal_size": round(won_deals[won_deals['Next Step'] == action]['Amount'].mean(), 2)
        }
        for action, count in next_step_analysis.head(10).items()
        if pd.notna(action)
    ]

def identify_missing_stakeholders() -> List[Dict]:
    """Identify deals missing key stakeholders"""
    
    df = opportunities_df.copy()
    
    # High-value open deals
    high_value_deals = df[
        (df['Amount'] > df['Amount'].quantile(0.75)) & 
        (~df['Stage'].isin(['Closed Won', 'Closed Lost']))
    ]
    
    # Check for missing stakeholders
    high_value_deals['contact_title_missing'] = high_value_deals['Contact: Title'].isna()
    
    missing_stakeholders = high_value_deals[high_value_deals['contact_title_missing']]
    
    return [
        {
            "account_name": row['Account Name'],
            "opportunity_name": row['Opportunity Name'],
            "amount": float(row['Amount']),
            "stage": row['Stage'],
            "current_contact": row['Primary Contact'] if pd.notna(row['Primary Contact']) else "No contact",
            "contact_title": row['Contact: Title'] if pd.notna(row['Contact: Title']) else "No title",
            "days_in_stage": int(row['Days in Stage'])
        }
        for _, row in missing_stakeholders.iterrows()
    ]

# PIPELINE HEALTH & EXECUTION FUNCTIONS

def analyze_stalled_opportunities(days_threshold: int = 30) -> List[Dict]:
    """Find opportunities stuck in Qualify or Propose stages"""
    
    df = opportunities_df.copy()
    
    stalled_opps = df[
        (df['Stage'].isin(['Qualify', 'Propose'])) & 
        (df['Days in Stage'] > days_threshold)
    ]
    
    return [
        {
            "account_name": row['Account Name'],
            "opportunity_name": row['Opportunity Name'],
            "stage": row['Stage'],
            "amount": float(row['Amount']),
            "probability": int(row['Probability (%)']),
            "days_in_stage": int(row['Days in Stage']),
            "next_step": row['Next Step'],
            "primary_contact": row['Primary Contact'],
            "contact_title": row['Contact: Title'],
            "urgency_level": "Critical" if row['Days in Stage'] > 60 else "High" if row['Days in Stage'] > 45 else "Medium"
        }
        for _, row in stalled_opps.iterrows()
    ]

def get_stage_metrics() -> List[Dict]:
    """Calculate average deal size and probability per stage"""
    
    df = opportunities_df.copy()
    
    stage_metrics = df.groupby('Stage').agg({
        'Amount': ['mean', 'median', 'sum', 'count'],
        'Probability (%)': ['mean', 'median'],
        'Days in Stage': 'mean'
    }).round(2)
    
    stage_metrics.columns = ['avg_deal_size', 'median_deal_size', 'total_value', 'deal_count', 'avg_probability', 'median_probability', 'avg_days_in_stage']
    
    return [
        {
            "stage": stage,
            "deal_count": int(row['deal_count']),
            "avg_deal_size": float(row['avg_deal_size']),
            "median_deal_size": float(row['median_deal_size']),
            "total_pipeline_value": float(row['total_value']),
            "avg_probability": round(float(row['avg_probability']), 1),
            "median_probability": round(float(row['median_probability']), 1),
            "avg_days_in_stage": round(float(row['avg_days_in_stage']), 1)
        }
        for stage, row in stage_metrics.iterrows()
    ]

def analyze_action_type_conversion() -> List[Dict]:
    """Analyze conversion rates by action type (Adoption, Implementation, etc.)"""
    
    df = opportunities_df.copy()
    concluded_deals = df[df['Stage'].isin(['Closed Won', 'Closed Lost'])]
    
    action_analysis = concluded_deals.groupby(['Action Type', 'Stage']).size().unstack(fill_value=0)
    
    results = []
    for action_type, row in action_analysis.iterrows():
        if action_type == 'Unknown':
            continue
            
        won = row.get('Closed Won', 0)
        lost = row.get('Closed Lost', 0)
        total = won + lost
        
        if total > 0:
            conversion_rate = (won / total) * 100
            avg_deal_size = concluded_deals[concluded_deals['Action Type'] == action_type]['Amount'].mean()
            avg_sales_cycle = concluded_deals[concluded_deals['Action Type'] == action_type]['Days in Stage'].mean()
            
            results.append({
                "action_type": action_type,
                "deals_won": int(won),
                "deals_lost": int(lost),
                "total_deals": int(total),
                "conversion_rate": round(conversion_rate, 1),
                "avg_deal_size": round(avg_deal_size, 2),
                "avg_sales_cycle_days": round(avg_sales_cycle, 1)
            })
    
    return sorted(results, key=lambda x: x['conversion_rate'], reverse=True)

def analyze_next_steps_by_outcome() -> Dict[str, Any]:
    """Analyze most common next steps in won vs lost deals"""
    
    df = opportunities_df.copy()
    
    won_deals = df[df['Stage'] == 'Closed Won']
    lost_deals = df[df['Stage'] == 'Closed Lost']
    
    won_next_steps = won_deals['Next Step'].value_counts().head(10)
    lost_next_steps = lost_deals['Next Step'].value_counts().head(10)
    
    # Create a combined list for visualization
    combined_data = []
    
    # Add winning next steps
    for step, count in won_next_steps.items():
        if pd.notna(step):
            combined_data.append({
                "next_step": step,
                "outcome": "Won",
                "frequency": int(count),
                "avg_deal_size": float(won_deals[won_deals['Next Step'] == step]['Amount'].mean())
            })
    
    # Add losing next steps
    for step, count in lost_next_steps.items():
        if pd.notna(step):
            combined_data.append({
                "next_step": step,
                "outcome": "Lost", 
                "frequency": int(count),
                "avg_deal_size": float(lost_deals[lost_deals['Next Step'] == step]['Amount'].mean())
            })
    
    return combined_data

# ADVANCED ANALYTICS FUNCTIONS

def get_comprehensive_pipeline_health() -> List[Dict[str, Any]]:
    """Comprehensive pipeline health analysis with visual insights"""
    
    df = opportunities_df.copy()
    
    # Analyze each pipeline stage for health metrics
    open_pipeline = df[~df['Stage'].isin(['Closed Won', 'Closed Lost'])]
    
    if len(open_pipeline) == 0:
        return [{"stage": "No Active Pipeline", "deal_count": 0, "total_value": 0, "health_score": 0}]
    
    # Group by stage and calculate health metrics
    stage_health = []
    
    for stage in open_pipeline['Stage'].unique():
        stage_deals = open_pipeline[open_pipeline['Stage'] == stage]
        
        # Calculate health indicators
        avg_probability = stage_deals['Probability (%)'].mean()
        avg_days_in_stage = stage_deals['Days in Stage'].mean()
        total_value = stage_deals['Amount'].sum()
        weighted_value = stage_deals['Weighted Value'].sum()
        deal_count = len(stage_deals)
        
        # Health score calculation (0-100)
        # Factors: probability, time in stage, value concentration
        probability_score = avg_probability  # 0-100
        velocity_score = max(0, 100 - (avg_days_in_stage - 15) * 2)  # Penalty after 15 days
        concentration_score = min(100, (deal_count / len(open_pipeline)) * 100 * 5)  # Healthy distribution
        
        health_score = (probability_score * 0.5 + velocity_score * 0.3 + concentration_score * 0.2)
        
        # Determine health status
        if health_score >= 75:
            health_status = "Excellent"
        elif health_score >= 60:
            health_status = "Good"
        elif health_score >= 40:
            health_status = "Fair"
        else:
            health_status = "Poor"
        
        stage_health.append({
            "stage": stage,
            "deal_count": deal_count,
            "total_value": float(total_value),
            "weighted_value": float(weighted_value),
            "avg_probability": float(avg_probability),
            "avg_days_in_stage": float(avg_days_in_stage),
            "health_score": float(health_score),
            "health_status": health_status
        })
    
    # Sort by total value descending
    return sorted(stage_health, key=lambda x: x['total_value'], reverse=True)

def get_account_360_view(account_name: str) -> Dict[str, Any]:
    """Comprehensive 360-degree view of a specific account"""
    
    # Opportunities for this account
    account_opps = opportunities_df[opportunities_df['Account Name'] == account_name]
    
    if len(account_opps) == 0:
        return {"error": f"No opportunities found for account: {account_name}"}
    
    # Contacts for this account
    account_contacts = contacts_df[contacts_df['Account Name'] == account_name]
    
    # Opportunity summary
    opp_summary = {
        "total_opportunities": len(account_opps),
        "total_value": round(account_opps['Amount'].sum(), 2),
        "weighted_value": round(account_opps['Weighted Value'].sum(), 2),
        "won_deals": len(account_opps[account_opps['Stage'] == 'Closed Won']),
        "lost_deals": len(account_opps[account_opps['Stage'] == 'Closed Lost']),
        "active_deals": len(account_opps[~account_opps['Stage'].isin(['Closed Won', 'Closed Lost'])])
    }
    
    # Stage distribution
    stage_dist = account_opps['Stage'].value_counts().to_dict()
    
    # Product preferences
    product_dist = account_opps['Product Category'].value_counts().to_dict()
    
    # Contact analysis
    contact_titles = account_contacts['Title'].value_counts().head(10).to_dict()
    
    # Recent activity
    recent_opps = account_opps.nlargest(5, 'Created Date')[
        ['Opportunity Name', 'Stage', 'Amount', 'Probability (%)', 'Next Step']
    ].to_dict('records')
    
    return {
        "account_name": account_name,
        "opportunity_summary": opp_summary,
        "stage_distribution": stage_dist,
        "product_preferences": product_dist,
        "contact_titles": contact_titles,
        "recent_opportunities": recent_opps,
        "total_contacts": len(account_contacts)
    }

# Trend analysis
def calculate_pipeline_trends(periods: int = 4) -> List[Dict[str, Any]]:
    """Calculate pipeline trends over time periods"""
    df = opportunities_df.copy()
    
    # Create time-based buckets
    df['Month'] = df['Created Date'].dt.to_period('M')
    
    # Get last N months
    latest_month = df['Month'].max()
    months = []
    for i in range(periods):
        months.append(latest_month - i)
    months.reverse()
    
    trends = []
    for month in months:
        month_data = df[df['Month'] == month]
        
        # Calculate metrics for the month
        total_pipeline = month_data[~month_data['Stage'].isin(['Closed Won', 'Closed Lost'])]['Amount'].sum()
        new_opportunities = len(month_data)
        avg_deal_size = month_data['Amount'].mean()
        win_rate = (len(month_data[month_data['Stage'] == 'Closed Won']) / 
                   len(month_data[month_data['Stage'].isin(['Closed Won', 'Closed Lost'])]) * 100) if len(month_data[month_data['Stage'].isin(['Closed Won', 'Closed Lost'])]) > 0 else 0
        
        trends.append({
            "period": str(month),
            "total_pipeline": float(total_pipeline),
            "new_opportunities": new_opportunities,
            "avg_deal_size": float(avg_deal_size),
            "win_rate": float(win_rate)
        })
    
    # Calculate trend direction
    if len(trends) >= 2:
        pipeline_change = ((trends[-1]['total_pipeline'] - trends[0]['total_pipeline']) / 
                          trends[0]['total_pipeline'] * 100) if trends[0]['total_pipeline'] > 0 else 0
        
        trends.append({
            "summary": {
                "pipeline_trend": "increasing" if pipeline_change > 0 else "decreasing",
                "pipeline_change_percent": round(pipeline_change, 1),
                "avg_monthly_pipeline": statistics.mean([t['total_pipeline'] for t in trends[:-1]]),
                "pipeline_volatility": statistics.stdev([t['total_pipeline'] for t in trends[:-1]]) if len(trends) > 2 else 0
            }
        })
    
    return trends

def compare_performance_metrics(dimension: str = "product_category", metric: str = "win_rate") -> List[Dict[str, Any]]:
    """Compare performance across different dimensions"""
    df = opportunities_df.copy()
    
    valid_dimensions = ['product_category', 'stage', 'type', 'account_name', 'service_type']
    valid_metrics = ['win_rate', 'avg_deal_size', 'total_value', 'avg_probability', 'avg_days_to_close']
    
    if dimension not in valid_dimensions:
        dimension = 'product_category'
    if metric not in valid_metrics:
        metric = 'win_rate'
    
    # Map dimension to actual column names
    dimension_map = {
        'product_category': 'Product Category',
        'stage': 'Stage',
        'type': 'Type',
        'account_name': 'Account Name',
        'service_type': 'Service Type'
    }
    
    actual_dimension = dimension_map.get(dimension, 'Product Category')
    
    results = []
    
    for category in df[actual_dimension].unique():
        if pd.isna(category):
            continue
            
        category_data = df[df[actual_dimension] == category]
        
        # Calculate requested metric
        if metric == 'win_rate':
            won = len(category_data[category_data['Stage'] == 'Closed Won'])
            total_closed = len(category_data[category_data['Stage'].isin(['Closed Won', 'Closed Lost'])])
            value = (won / total_closed * 100) if total_closed > 0 else 0
        elif metric == 'avg_deal_size':
            value = category_data['Amount'].mean()
        elif metric == 'total_value':
            value = category_data['Amount'].sum()
        elif metric == 'avg_probability':
            value = category_data['Probability (%)'].mean()
        elif metric == 'avg_days_to_close':
            closed_deals = category_data[category_data['Stage'].isin(['Closed Won', 'Closed Lost'])]
            value = closed_deals['Days in Stage'].mean() if len(closed_deals) > 0 else 0
        
        results.append({
            "category": str(category),
            "metric_name": metric,
            "metric_value": float(value),
            "sample_size": len(category_data),
            "dimension": dimension
        })
    
    # Sort by metric value
    results = sorted(results, key=lambda x: x['metric_value'], reverse=True)
    
    # Add comparative analysis
    if results:
        values = [r['metric_value'] for r in results]
        avg_value = statistics.mean(values)
        
        for result in results:
            result['vs_average'] = round(((result['metric_value'] - avg_value) / avg_value * 100), 1) if avg_value > 0 else 0
            result['performance'] = "above average" if result['metric_value'] > avg_value else "below average"
    
    return results

def analyze_deal_velocity() -> Dict[str, Any]:
    """Analyze how quickly deals move through the pipeline"""
    df = opportunities_df.copy()
    
    # Calculate velocity metrics by stage
    velocity_by_stage = df.groupby('Stage').agg({
        'Days in Stage': ['mean', 'median', 'std'],
        'Amount': 'count'
    }).round(1)
    
    velocity_by_stage.columns = ['avg_days', 'median_days', 'std_dev', 'deal_count']
    
    # Calculate overall pipeline velocity
    closed_deals = df[df['Stage'].isin(['Closed Won', 'Closed Lost'])]
    
    overall_metrics = {
        "avg_sales_cycle": float(closed_deals['Days in Stage'].mean()) if len(closed_deals) > 0 else 0,
        "fastest_deal_days": int(closed_deals['Days in Stage'].min()) if len(closed_deals) > 0 else 0,
        "slowest_deal_days": int(closed_deals['Days in Stage'].max()) if len(closed_deals) > 0 else 0,
        "deals_over_60_days": len(df[(df['Days in Stage'] > 60) & (~df['Stage'].isin(['Closed Won', 'Closed Lost']))]),
        "velocity_by_stage": velocity_by_stage.to_dict('index')
    }
    
    # Identify bottlenecks
    bottlenecks = []
    for stage, metrics in overall_metrics['velocity_by_stage'].items():
        if metrics['avg_days'] > 30 and stage not in ['Closed Won', 'Closed Lost']:
            bottlenecks.append({
                "stage": stage,
                "avg_days": metrics['avg_days'],
                "deals_stuck": metrics['deal_count']
            })
    
    overall_metrics['bottlenecks'] = sorted(bottlenecks, key=lambda x: x['avg_days'], reverse=True)
    
    return overall_metrics

# FUNCTION REGISTRY AND OPENAI DEFINITIONS

FUNCTION_HANDLERS = {
    # Performance & Forecasting
    "get_top_opportunities_by_value": get_top_opportunities_by_value,
    "get_high_probability_revenue_forecast": get_high_probability_revenue_forecast,
    "analyze_product_win_rates": analyze_product_win_rates,
    "analyze_win_rates_by_type": analyze_win_rates_by_type,
    "analyze_fiscal_period_distribution": analyze_fiscal_period_distribution,
    
    # Account & Opportunity Insights
    "get_top_accounts_by_total_value": get_top_accounts_by_total_value,
    "get_expansion_opportunities": get_expansion_opportunities,
    "analyze_stalled_high_stage_deals": analyze_stalled_high_stage_deals,
    "analyze_overdue_next_steps": analyze_overdue_next_steps,
    "analyze_recent_losses": analyze_recent_losses,
    
    # Contact & Role Influence
    "analyze_winning_contact_titles": analyze_winning_contact_titles,
    "analyze_c_level_engagement": analyze_c_level_engagement,
    "analyze_successful_cro_actions": analyze_successful_actions,
    "identify_missing_stakeholders": identify_missing_stakeholders,
    
    # Pipeline Health & Execution
    "analyze_stalled_opportunities": analyze_stalled_opportunities,
    "get_stage_metrics": get_stage_metrics,
    "analyze_action_type_conversion": analyze_action_type_conversion,
    "analyze_next_steps_by_outcome": analyze_next_steps_by_outcome,
    
    # Advanced Analytics
    "get_comprehensive_pipeline_health": get_comprehensive_pipeline_health,
    "get_account_360_view": get_account_360_view,
    "calculate_pipeline_trends": calculate_pipeline_trends,
    "compare_performance_metrics": compare_performance_metrics,
    "analyze_deal_velocity": analyze_deal_velocity
}

FUNCTIONS = [
    {
        "name": "get_top_opportunities_by_value",
        "description": "Get top opportunities by deal value with stage and probability information",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "number", "description": "Number of top opportunities to return"},
                "stage_filter": {"type": "string", "description": "Filter by specific stage"},
                "include_probability": {"type": "boolean", "description": "Include probability in analysis"}
            }
        }
    },
    {
        "name": "get_high_probability_revenue_forecast",
        "description": "Calculate expected revenue from high-probability deals",
        "parameters": {
            "type": "object",
            "properties": {
                "min_probability": {"type": "number", "description": "Minimum probability threshold (default: 70)"},
                "fiscal_period": {"type": "string", "description": "Specific fiscal period to analyze"}
            }
        }
    },
    {
        "name": "analyze_product_win_rates",
        "description": "Analyze win rates by product/service categories",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "analyze_win_rates_by_type",
        "description": "Analyze win rates by opportunity type (New Business, Expansion, Renewal)",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "analyze_fiscal_period_distribution",
        "description": "Analyze deal distribution and crowding across fiscal periods",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "get_top_accounts_by_total_value",
        "description": "Get top accounts by total opportunity value (won + open)",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "number", "description": "Number of top accounts to return"},
                "include_closed_won": {"type": "boolean", "description": "Include closed deals in calculation"},
                "include_closed_lost": {"type": "boolean", "description": "Include closed lost deals in calculation. Generally False by default"}
            }
        }
    },
    {
        "name": "get_expansion_opportunities",
        "description": "Get accounts with open Expansion opportunities and their current stages",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "analyze_stalled_high_stage_deals",
        "description": "Find accounts that reach Negotiate/Propose but end up as Closed Lost",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "analyze_overdue_next_steps",
        "description": "Find opportunities with overdue or unclear next steps",
        "parameters": {
            "type": "object",
            "properties": {
                "days_threshold": {"type": "number", "description": "Days threshold for considering overdue"}
            }
        }
    },
    {
        "name": "analyze_recent_losses",
        "description": "Analyze recently lost deals and their characteristics",
        "parameters": {
            "type": "object",
            "properties": {
                "days_back": {"type": "number", "description": "Number of days back to analyze losses"}
            }
        }
    },
    {
        "name": "analyze_winning_contact_titles",
        "description": "Analyze which contact titles appear most often in Closed Won deals",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "analyze_c_level_engagement",
        "description": "Analyze C-level engagement in late-stage opportunities",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "analyze_successful_actions",
        "description": "Analyze actions associated with successful deals",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "identify_missing_stakeholders",
        "description": "Identify high-value deals missing key stakeholders",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "analyze_stalled_opportunities",
        "description": "Find opportunities stuck in Qualify or Propose stages for extended periods",
        "parameters": {
            "type": "object",
            "properties": {
                "days_threshold": {"type": "number", "description": "Days threshold for considering stalled"}
            }
        }
    },
    {
        "name": "get_stage_metrics",
        "description": "Calculate average deal size, probability, and velocity per stage",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "analyze_action_type_conversion",
        "description": "Analyze conversion rates by action type (Adoption Initiative, Implementation, etc.)",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "analyze_next_steps_by_outcome",
        "description": "Analyze most common next steps in won vs lost deals",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "get_comprehensive_pipeline_health",
        "description": "Get comprehensive pipeline health analysis and metrics",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "get_account_360_view",
        "description": "Get comprehensive 360-degree view of a specific account",
        "parameters": {
            "type": "object",
            "properties": {
                "account_name": {"type": "string", "description": "Name of the account to analyze"}
            },
            "required": ["account_name"]
        }
    },
    {
        "name": "calculate_pipeline_trends",
        "description": "Calculate pipeline trends over time periods with statistical analysis",
        "parameters": {
            "type": "object",
            "properties": {
                "periods": {"type": "number", "description": "Number of time periods to analyze (default: 4)"}
            }
        }
    },
    {
        "name": "compare_performance_metrics",
        "description": "Compare performance metrics across different dimensions (products, stages, types, accounts)",
        "parameters": {
            "type": "object",
            "properties": {
                "dimension": {
                    "type": "string", 
                    "description": "Dimension to compare: product_category, stage, type, account_name, service_type"
                },
                "metric": {
                    "type": "string",
                    "description": "Metric to compare: win_rate, avg_deal_size, total_value, avg_probability, avg_days_to_close"
                }
            }
        }
    },
    {
        "name": "analyze_deal_velocity",
        "description": "Analyze how quickly deals move through the pipeline and identify bottlenecks",
        "parameters": {"type": "object", "properties": {}}
    }
]

# Data summarization functions using LLM
async def summarize_function_results(function_results: Dict[str, Any], original_question: str) -> List[SummarizedData]:
    """Use LLM to create presentation-ready visualizations from raw function results"""
    
    summarization_prompt = f"""
You are a data visualization expert for executive dashboards. Your task is to convert raw analytics function results into presentation-ready visualizations for a CRO dashboard.

ORIGINAL QUESTION: {original_question}

AVAILABLE FUNCTION RESULTS:
{json.dumps(function_results, indent=2)}

Your task is to create 1-3 optimal visualizations that best answer the user's question. For each visualization, determine:

1. **Chart Type Selection**:
   - 'metrics': For summary numbers, KPIs, totals (use when you have 1-6 key numbers)
   - 'bar': For comparing categories, rankings, or discrete values
   - 'pie': For showing parts of a whole (max 8 segments, use percentages)
   - 'line': For trends over time or sequential data
   - 'table': For detailed data that needs full context (use sparingly)

2. **Data Preparation**:
   - Limit bar/pie charts to 10-15 items max for readability
   - Format numbers appropriately (currency, percentages)
   - Use clear, descriptive labels
   - Sort data meaningfully (highest to lowest, alphabetical, etc.)

3. **Insights Generation**:
   - Identify 2-4 key insights from the data
   - Focus on actionable findings for a CRO
   - Highlight risks, opportunities, or notable patterns

4. **Data Sources**:
   - Always include "data_sources" array with relevant source files
   - For opportunity data: ["opportunities.csv"]
   - For contact data: ["account_and_contact.csv"] 
   - For combined analysis: ["opportunities.csv", "account_and_contact.csv"]

RESPONSE FORMAT (JSON array):
[
  {{
    "title": "Clear, descriptive title for the visualization",
    "description": "Brief explanation of what this shows",
    "chart_type": "metrics|bar|pie|line|table",
    "data": [
      // For 'metrics': {{"label": "Total Pipeline", "value": 5400000, "format": "currency"}}
      // For 'bar/pie': {{"name": "Category", "value": 1000, "formatted": "$1,000"}}
      // For 'line': {{"period": "Q1", "revenue": 500000, "deals": 25}}
      // For 'table': Full objects with all relevant fields
    ],
    "key_insights": [
      "Insight 1: specific finding with numbers",
      "Insight 2: actionable recommendation"
    ],
    "source_functions": ["function_name_1", "function_name_2"],
    "data_sources": ["opportunities.csv", "account_and_contact.csv"]
  }}
]

IMPORTANT GUIDELINES:
- Prioritize the most relevant visualizations for the question asked
- Keep data concise but informative
- Format all monetary values as currency
- Use clear, business-friendly language
- Focus on executive-level insights
- If multiple functions were called, create separate visualizations or combine related data intelligently
- ALWAYS include data_sources array indicating which CSV files the data comes from
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data visualization expert. Respond only with valid JSON array as specified."},
                {"role": "user", "content": summarization_prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        response_content = completion.choices[0].message.content.strip()
        
        # Clean up the response to ensure it's valid JSON
        if response_content.startswith('```json'):
            response_content = response_content[7:-3]
        elif response_content.startswith('```'):
            response_content = response_content[3:-3]
        
        summarized_data = json.loads(response_content)
        
        return [SummarizedData(**item) for item in summarized_data]
        
    except Exception as e:
        print(f"❌ Error in summarization: {e}")
        # Fallback to simple conversion
        return create_fallback_visualizations(function_results)

def create_fallback_visualizations(function_results: Dict[str, Any]) -> List[SummarizedData]:
    """Fallback visualization creation if LLM summarization fails"""
    visualizations = []
    
    for func_name, data in function_results.items():
        # Simple heuristic for chart type
        if isinstance(data, list) and len(data) > 0:
            chart_type = "table" if len(data) > 15 else "bar"
        elif isinstance(data, dict):
            chart_type = "metrics"
        else:
            chart_type = "table"

        # Determine data sources based on function name
        data_sources = []
        if any(keyword in func_name.lower() for keyword in ['account', 'contact', 'stakeholder', 'title']):
            data_sources = ["opportunities.csv", "account_and_contact.csv"]
        else:
            data_sources = ["opportunities.csv"]
        
        viz = SummarizedData(
            title=func_name.replace('_', ' ').title(),
            description=f"Results from {func_name}",
            chart_type=chart_type,
            data=data if isinstance(data, list) else [data] if isinstance(data, dict) else [],
            key_insights=[f"Data from {func_name} analysis"],
            source_functions=[func_name],
            data_sources=data_sources
        )
        visualizations.append(viz)
    
    return visualizations

# Enhanced error handling function
def create_error_response(question: str, error_type: str = "unclear") -> str:
    """Create helpful error responses for different error types"""
    
    error_responses = {
        "unclear": f"""I understand you're asking about "{question}", but I need more clarity to provide accurate insights.

Could you please specify:
- Which metric you're interested in (revenue, win rate, pipeline value, etc.)?
- The time period or scope (all time, specific quarter, specific accounts)?
- What type of analysis you need (comparison, trend, breakdown)?

For example:
- "What's our win rate by product category?"
- "Show me pipeline trends over the last 4 months"
- "Compare average deal sizes across opportunity types"

I have access to detailed opportunity and contact data and can analyze:
- Pipeline metrics and forecasts
- Win rates and conversion analysis
- Account and contact insights
- Deal velocity and bottlenecks
- Product performance comparisons""",
        
        "no_data": f"""I couldn't find sufficient data to answer "{question}".

This might be because:
1. The specific data you're looking for doesn't exist in our dataset
2. The filters applied resulted in no matching records
3. The time period specified has no data

Available data includes:
- {len(opportunities_df)} opportunity records
- {len(contacts_df)} contact records
- Stages: {', '.join(opportunities_df['Stage'].unique())}
- Product Categories: {', '.join([str(x) for x in opportunities_df['Product Category'].unique() if pd.notna(x)][:5])}...

Please try rephrasing your question or asking about available data dimensions.""",
        
        "invalid_params": """The parameters provided seem to be invalid or out of range.

Please ensure:
- Date ranges are valid and within our data period
- Numeric values are positive where required
- Category names match our data (case-sensitive)
- Percentages are between 0 and 100

You can ask me to list available categories, stages, or other dimensions if you're unsure."""
    }
    
    return error_responses.get(error_type, error_responses["unclear"])

async def extract_key_metrics(function_results: Dict[str, Any], original_question: str) -> Dict[str, Any]:
    """Extract key metrics summary using LLM"""
    
    metrics_prompt = f"""
Extract 3-6 key executive metrics from the analytics results to display as KPI cards.

ORIGINAL QUESTION: {original_question}
FUNCTION RESULTS: {json.dumps(function_results, indent=2)}

Return a JSON object with key metrics that a CRO would want to see at a glance:

{{
  "total_pipeline_value": {{"value": 5400000, "label": "Total Pipeline", "format": "currency", "trend": "up"}},
  "win_rate": {{"value": 23.5, "label": "Overall Win Rate", "format": "percentage", "trend": "stable"}},
  "avg_deal_size": {{"value": 125000, "label": "Avg Deal Size", "format": "currency", "trend": "down"}},
  "high_prob_deals": {{"value": 12, "label": "High Probability Deals", "format": "number", "trend": "up"}}
}}

Focus on:
- Revenue/pipeline numbers
- Conversion rates and percentages  
- Deal counts and averages
- Performance indicators

Use "currency", "percentage", or "number" for format. Use "up", "down", or "stable" for trend.
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Extract key metrics as JSON object only."},
                {"role": "user", "content": metrics_prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )
        
        response_content = completion.choices[0].message.content.strip()
        
        # Clean up JSON
        if response_content.startswith('```json'):
            response_content = response_content[7:-3]
        elif response_content.startswith('```'):
            response_content = response_content[3:-3]
        
        return json.loads(response_content)
        
    except Exception as e:
        print(f"❌ Error extracting metrics: {e}")
        return {}
    
# API Routes (Enhanced)
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with enhanced metrics"""
    return HealthResponse(
        status="healthy",
        data_loaded=opportunities_df is not None and contacts_df is not None,
        timestamp=datetime.now().isoformat(),
        records_count={
            "opportunities": len(opportunities_df) if opportunities_df is not None else 0,
            "contacts": len(contacts_df) if contacts_df is not None else 0,
            "active_opportunities": len(opportunities_df[~opportunities_df['Stage'].isin(['Closed Won', 'Closed Lost'])]) if opportunities_df is not None else 0,
            "won_opportunities": len(opportunities_df[opportunities_df['Stage'] == 'Closed Won']) if opportunities_df is not None else 0
        }
    )

@app.get("/api/data/schema")
async def get_data_schema():
    """Get enhanced data schema information"""
    if opportunities_df is None or contacts_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    return {
        "opportunities": {
            "count": len(opportunities_df),
            "columns": opportunities_df.columns.tolist(),
            "sample": opportunities_df.head(1).to_dict('records')[0] if len(opportunities_df) > 0 else {},
            "stages": opportunities_df['Stage'].unique().tolist(),
            "product_categories": opportunities_df['Product Category'].unique().tolist(),
            "fiscal_periods": opportunities_df['Fiscal Period'].unique().tolist()
        },
        "contacts": {
            "count": len(contacts_df),
            "columns": contacts_df.columns.tolist(),
            "sample": contacts_df.head(1).to_dict('records')[0] if len(contacts_df) > 0 else {},
            "top_titles": contacts_df['Title'].value_counts().head(10).to_dict()
        }
    }

@app.get("/api/quick-insights")
async def get_quick_insights():
    """Get quick dashboard insights"""
    if opportunities_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = opportunities_df.copy()
    
    # Calculate key metrics
    total_pipeline = df[~df['Stage'].isin(['Closed Won', 'Closed Lost'])]['Amount'].sum()
    weighted_pipeline = df[~df['Stage'].isin(['Closed Won', 'Closed Lost'])]['Weighted Value'].sum()
    won_deals = len(df[df['Stage'] == 'Closed Won'])
    total_concluded = len(df[df['Stage'].isin(['Closed Won', 'Closed Lost'])])
    win_rate = (won_deals / total_concluded * 100) if total_concluded > 0 else 0
    
    return {
        "total_pipeline_value": round(total_pipeline, 2),
        "weighted_pipeline_value": round(weighted_pipeline, 2),
        "overall_win_rate": round(win_rate, 1),
        "active_deals": len(df[~df['Stage'].isin(['Closed Won', 'Closed Lost'])]),
        "deals_in_negotiate": len(df[df['Stage'] == 'Negotiate']),
        "high_probability_deals": len(df[(df['Probability (%)'] > 70) & (~df['Stage'].isin(['Closed Won', 'Closed Lost']))]),
        "stalled_deals": len(df[(df['Days in Stage'] > 30) & (df['Stage'].isin(['Qualify', 'Propose']))])
    }


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process queries with LLM-summarized, presentation-ready responses"""
    print("Query received")
    if opportunities_df is None or contacts_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    try:
        # Enhanced system prompt
        system_prompt = f"""You are an expert CRO analytics assistant. You analyze enterprise sales data and provide executive-level insights.

BUSINESS CONTEXT:
- B2B enterprise sales pipeline analysis
- Products: SaaS (ERP, CX, HCM), PaaS (Analytics, AI), OCI (Compute, Storage)
- Deal types: New Business, Expansion, Renewal
- Sales stages: Qualify → Meet & Present → Propose → Negotiate → Closed Won/Lost

AVAILABLE DATA:
- Opportunities: {len(opportunities_df)} records
- Contacts: {len(contacts_df)} records

ANALYSIS APPROACH:
1. Call relevant functions to gather data
2. The system will automatically summarize results into executive dashboards
3. Focus on providing strategic insights and recommendations
4. Call 2-4 functions for comprehensive analysis when needed

Your function results will be automatically converted into executive-friendly visualizations."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.question}
        ]
        
        max_iterations = request.max_iterations or 5
        functions_used = []
        all_function_results = {}
        iteration_count = 0
        
        # Multi-turn conversation loop (same as before)
        while iteration_count < max_iterations:
            iteration_count += 1
            
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                functions=FUNCTIONS,
                function_call="auto",
                temperature=0.1,
            )
            
            message = completion.choices[0].message
            
            messages.append({
                "role": "assistant",
                "content": message.content,
                "function_call": message.function_call
            })
            
            if message.function_call:
                function_name = message.function_call.name
                function_args = json.loads(message.function_call.arguments)
                
                if function_name in FUNCTION_HANDLERS:
                    function_result = FUNCTION_HANDLERS[function_name](**function_args)
                    function_result = convert_numpy_types(function_result)
                    
                    functions_used.append(function_name)
                    all_function_results[function_name] = function_result
                    
                    print(f"✅ Iteration {iteration_count}: Function '{function_name}' executed")
                    
                    messages.append({
                        "role": "function",
                        "name": function_name,
                        "content": json.dumps(function_result)
                    })
                    
                    if iteration_count < max_iterations:
                        reflection_prompt = """Based on the function result, consider if additional analysis would provide more comprehensive insights. If you have sufficient data to answer the user's question thoroughly, provide your final analysis."""
                        
                        messages.append({
                            "role": "system",
                            "content": reflection_prompt
                        })
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown function: {function_name}")
            else:
                break
        
        # Generate final executive response
        if functions_used:
            final_prompt = f"""Provide a comprehensive executive summary based on your analysis. Focus on:
1. Key findings with specific numbers
2. Strategic implications for revenue
3. Actionable recommendations
4. Risk and opportunity identification

Be concise but thorough - this is for C-level consumption."""
            
            messages.append({"role": "system", "content": final_prompt})
            
            final_completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.1
            )
            
            final_answer = final_completion.choices[0].message.content
        else:
            final_answer = message.content
        
        # Create summarized visualizations and metrics using LLM
        visualizations = []
        key_metrics = {}
        
        if all_function_results:
            # Create presentation-ready visualizations
            visualizations = await summarize_function_results(all_function_results, request.question)
            
            # Extract key metrics
            key_metrics = await extract_key_metrics(all_function_results, request.question)

            # Collect all unique data sources
            all_sources = set()
            for viz in visualizations:
                if viz.data_sources:
                    all_sources.update(viz.data_sources)
            overall_data_sources = list(all_sources)
        
        # Create execution summary
        execution_summary = None
        if functions_used:
            execution_summary = f"Executed {len(functions_used)} analytics functions across {iteration_count} iterations: {', '.join(functions_used)}"
        
        return QueryResponse(
            answer=final_answer,
            visualizations=visualizations,
            key_metrics=key_metrics,
            conversation_id=request.conversation_id,
            functions_executed=functions_used,
            execution_summary=execution_summary,
            data_sources=overall_data_sources
        )
            
    except Exception as e:
        print(f"❌ Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

# Streaming function to only yield final response
async def stream_query_response(request: StreamingQueryRequest) -> AsyncGenerator[str, None]:
    """Stream query response with context awareness"""
    
    if opportunities_df is None or contacts_df is None:
        yield f"data: {json.dumps({'error': 'Data not loaded'})}\n\n"
        return
    
    try:
        # Get or create conversation context
        conversation_id = request.conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if conversation_id not in conversation_contexts:
            conversation_contexts[conversation_id] = {
                'history': deque(maxlen=10),  # Keep last 10 exchanges
                'entities': set(),  # Track mentioned entities
                'metrics': set(),  # Track discussed metrics
                'last_function_results': {}
            }
        
        context = conversation_contexts[conversation_id]
        
        # Add current question to context
        context['history'].append({
            'role': 'user',
            'content': request.question,
            'timestamp': datetime.now().isoformat()
        })
        
        # Enhanced system prompt with context
        context_summary = ""
        if len(context['history']) > 1:
            recent_topics = list(context['entities'])[-5:] if context['entities'] else []
            recent_metrics = list(context['metrics'])[-5:] if context['metrics'] else []
            
            context_summary = f"""
CONVERSATION CONTEXT:
- Recent topics discussed: {', '.join(recent_topics) if recent_topics else 'None'}
- Recent metrics analyzed: {', '.join(recent_metrics) if recent_metrics else 'None'}
- Previous questions: {len(context['history']) - 1}

When answering, consider if this is a follow-up question and maintain context from previous exchanges.
"""
        
        system_prompt = f"""You are an expert CRO analytics assistant. You analyze enterprise sales data and provide executive-level insights.

BUSINESS CONTEXT:
- B2B enterprise sales pipeline analysis
- Products: SaaS (ERP, CX, HCM), PaaS (Analytics, AI), OCI (Compute, Storage)
- Deal types: New Business, Expansion, Renewal
- Sales stages: Qualify → Meet & Present → Propose → Negotiate → Closed Won/Lost

AVAILABLE DATA:
- Opportunities: {len(opportunities_df)} records from opportunities.csv
- Contacts: {len(contacts_df)} records from account_and_contact.csv

{context_summary}

ANALYSIS APPROACH:
1. Understand if this is a follow-up question and maintain context
2. Call relevant functions to gather data
3. Handle unclear questions by asking for clarification
4. Provide strategic insights and recommendations
5. Use advanced analytics functions for trends and comparisons when appropriate

ERROR HANDLING:
- If a question is unclear, provide helpful guidance on how to rephrase
- If no data is found, explain what data is available
- Always be helpful and suggest alternatives"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.question}
        ]
        
        # Add context from recent history if relevant
        if len(context['history']) > 1 and "previous" in request.question.lower() or "last" in request.question.lower():
            messages.insert(1, {
                "role": "system", 
                "content": f"Previous question context: {context['history'][-2]['content'] if len(context['history']) > 1 else 'None'}"
            })
        
        max_iterations = request.max_iterations or 5
        functions_used = []
        all_function_results = {}
        iteration_count = 0
        has_valid_results = False
        
        # Store conversation for PDF generation
        conversation_history[conversation_id] = {
            'query': request.question,
            'timestamp': datetime.now().isoformat(),
            'functions_executed': [],
            'visualizations': [],
            'key_metrics': {},
            'data_sources': []
        }
        
        # Multi-turn conversation loop
        while iteration_count < max_iterations:
            iteration_count += 1
            
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                functions=FUNCTIONS,
                function_call="auto",
                temperature=0.1,
            )
            
            message = completion.choices[0].message
            
            messages.append({
                "role": "assistant",
                "content": message.content,
                "function_call": message.function_call
            })
            
            if message.function_call:
                function_name = message.function_call.name
                function_args = json.loads(message.function_call.arguments)
                
                if function_name in FUNCTION_HANDLERS:
                    try:
                        function_result = FUNCTION_HANDLERS[function_name](**function_args)
                        function_result = convert_numpy_types(function_result)
                        
                        # Check if we got valid results
                        if function_result and (isinstance(function_result, list) and len(function_result) > 0 or 
                                              isinstance(function_result, dict) and function_result):
                            has_valid_results = True
                        
                        functions_used.append(function_name)
                        all_function_results[function_name] = function_result
                        
                        # Update context with results
                        context['last_function_results'] = all_function_results
                        
                        # Extract entities and metrics from results
                        if isinstance(function_result, list):
                            for item in function_result[:5]:  # Sample first 5
                                if isinstance(item, dict):
                                    if 'account_name' in item:
                                        context['entities'].add(item['account_name'])
                                    if 'product_category' in item:
                                        context['entities'].add(item['product_category'])
                        
                        print(f"✅ Iteration {iteration_count}: Function '{function_name}' executed")
                        
                        messages.append({
                            "role": "function",
                            "name": function_name,
                            "content": json.dumps(function_result)
                        })
                        
                        if iteration_count < max_iterations:
                            reflection_prompt = """Based on the function result, consider if additional analysis would provide more comprehensive insights. If you have sufficient data to answer the user's question thoroughly, provide your final analysis."""
                            
                            messages.append({
                                "role": "system",
                                "content": reflection_prompt
                            })
                    except Exception as func_error:
                        print(f"Function {function_name} failed: {func_error}")
                        messages.append({
                            "role": "function",
                            "name": function_name,
                            "content": json.dumps({"error": str(func_error)})
                        })
                else:
                    yield f"data: {json.dumps({'error': f'Unknown function: {function_name}'})}\n\n"
                    return
            else:
                break
        
        # Generate final response with error handling
        if functions_used:
            # Check if we got meaningful results
            if not has_valid_results:
                error_response = create_error_response(request.question, "no_data")
                final_answer = error_response
            else:
                final_prompt = f"""Provide a comprehensive executive summary based on your analysis. 

If the data seems insufficient or the question needs clarification, provide helpful guidance.

Focus on:
1. Key findings with specific numbers
2. Strategic implications for revenue
3. Actionable recommendations
4. Risk and opportunity identification

Be concise but thorough - this is for C-level consumption."""
                
                messages.append({"role": "system", "content": final_prompt})
                
                final_completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.1
                )
                
                final_answer = final_completion.choices[0].message.content
        else:
            # No functions called - likely an unclear question
            if "error" in message.content.lower() or not message.content:
                final_answer = create_error_response(request.question, "unclear")
            else:
                final_answer = message.content
        
        # Update conversation context
        context['history'].append({
            'role': 'assistant',
            'content': final_answer,
            'functions_used': functions_used,
            'timestamp': datetime.now().isoformat()
        })
        
        # Create summarized visualizations and metrics
        visualizations = []
        key_metrics = {}
        overall_data_sources = []
        
        if all_function_results and has_valid_results:
            visualizations = await summarize_function_results(all_function_results, request.question)
            key_metrics = await extract_key_metrics(all_function_results, request.question)
            
            all_sources = set()
            for viz in visualizations:
                if viz.data_sources:
                    all_sources.update(viz.data_sources)
            overall_data_sources = list(all_sources)
        
        # Update conversation history
        conversation_history[conversation_id].update({
            'answer': final_answer,
            'functions_executed': functions_used,
            'visualizations': [viz.dict() for viz in visualizations],
            'key_metrics': key_metrics,
            'data_sources': overall_data_sources
        })
        
        execution_summary = None
        if functions_used:
            execution_summary = f"Executed {len(functions_used)} analytics functions: {', '.join(functions_used)}"
        
        final_response = {
            'status': 'complete',
            'answer': final_answer,
            'visualizations': [viz.dict() for viz in visualizations],
            'key_metrics': key_metrics,
            'conversation_id': conversation_id,
            'functions_executed': functions_used,
            'execution_summary': execution_summary,
            'data_sources': overall_data_sources,
            'streaming_complete': True,
            'has_context': len(context['history']) > 2
        }
        
        yield f"data: {json.dumps(final_response)}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'error': f'Query processing failed: {str(e)}'})}\n\n"

# New API Routes for Streaming and PDF

@app.post("/api/query/stream")
async def stream_query(request: StreamingQueryRequest):
    """Stream query response with real-time updates"""
    
    async def generate():
        async for chunk in stream_query_response(request):
            yield chunk
        # Send final completion marker
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

@app.post("/api/report/pdf")
async def generate_pdf_report_endpoint(request: PDFReportRequest):
    """Generate PDF report from conversation history"""
    
    if request.conversation_id not in conversation_history:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    try:
        conversation_data = conversation_history[request.conversation_id]
        
        # Generate PDF
        pdf_bytes = generate_pdf_report(conversation_data, request.title)
        
        # Create filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"revenue_analytics_report_{timestamp}.pdf"
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(pdf_bytes))
            }
        )
        
    except Exception as e:
        print(f"❌ PDF generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

@app.get("/api/conversations")
async def get_conversations():
    """Get list of available conversations for PDF generation"""
    
    conversations = []
    for conv_id, data in conversation_history.items():
        conversations.append({
            'conversation_id': conv_id,
            'timestamp': data.get('timestamp'),
            'query': data.get('query', '')[:100] + '...' if len(data.get('query', '')) > 100 else data.get('query', ''),
            'functions_count': len(data.get('functions_executed', [])),
            'visualizations_count': len(data.get('visualizations', []))
        })
    
    return {
        'conversations': sorted(conversations, key=lambda x: x['timestamp'], reverse=True)
    }

@app.get("/")
async def root():
    """Enhanced API documentation"""
    return {
        "message": "Enhanced Revenue Analytics API v2.1",
        "description": "CRO analytics with LLM-powered summarized visualizations",
        "version": "2.1.0",
        "features": {
            "llm_summarization": "Function results converted to presentation-ready visualizations",
            "executive_metrics": "Automatic extraction of key KPIs",
            "smart_charts": "Optimal chart type selection based on data",
            "multi_function_analysis": "Comprehensive cross-functional insights"
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/api/health", 
            "query": "/api/query"
        }
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load and prepare data when the application starts"""
    print("🚀 Starting Enhanced Revenue Analytics Server v2.0...")
    print("📊 Loading and processing CSV data...")
    load_csv_data()
    print("✅ Enhanced analytics server ready with 16 specialized functions!")
    print("🎯 Ready to answer comprehensive CRO questions!")

if __name__ == "__main__":
    print("\n🎯 Enhanced Revenue Analytics FastAPI Server v2.0")
    print("📈 Comprehensive CRO Analytics System")
    print("🔧 Loading data and starting server...")
    
    # Load data first
    load_csv_data()
    
    # Start server
    uvicorn.run(
        "enhanced_analytics_server:app",
        host="0.0.0.0",
        port=8083,
        reload=True,
        log_level="info"
    )