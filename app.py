import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Telemarketing Campaign Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .download-btn {
        background-color: #10B981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        border: none;
        cursor: pointer;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

def load_data(uploaded_file):
    """Load and clean the transaction data"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        st.success(f"✅ Successfully loaded {len(df):,} transactions")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Check required columns
        required_columns = ['User Identifier', 'Product Name', 'Service Name', 
                           'Created At', 'Entity Name', 'Full Name']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.info(f"Available columns: {list(df.columns)}")
            return None
        
        # Data cleaning
        df['Product Name'] = df['Product Name'].astype(str).str.strip()
        df['Entity Name'] = df['Entity Name'].astype(str).str.strip()
        df['User Identifier'] = pd.to_numeric(df['User Identifier'], errors='coerce')
        
        # Parse date column with multiple format handling
        df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce')
        
        # Add derived columns
        df['Date'] = df['Created At'].dt.date
        df['Day'] = df['Created At'].dt.day_name()
        df['Hour'] = df['Created At'].dt.hour
        
        return df
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def filter_data(df, start_date, end_date, product_filter, customer_type_filter):
    """Filter data based on selected criteria"""
    filtered_df = df.copy()
    
    # Date filter
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['Created At'] >= pd.Timestamp(start_date)) & 
            (filtered_df['Created At'] <= pd.Timestamp(end_date + timedelta(days=1)))
        ]
    
    # Product filter
    if product_filter:
        if product_filter != 'All':
            filtered_df = filtered_df[filtered_df['Product Name'].str.contains(product_filter, case=False, na=False)]
    
    # Customer type filter
    if customer_type_filter and customer_type_filter != 'All':
        if customer_type_filter == 'Customer Only':
            filtered_df = filtered_df[
                (filtered_df['Entity Name'].str.contains('Customer', case=False, na=False)) |
                (filtered_df['Entity Name'].isna()) |
                (filtered_df['Entity Name'] == '')
            ]
        elif customer_type_filter == 'Vendor/Agent Only':
            filtered_df = filtered_df[
                (filtered_df['Entity Name'].str.contains('Vendor|Agent', case=False, na=False))
            ]
    
    return filtered_df

def create_visualizations(filtered_df, start_date, end_date):
    """Create all visualizations"""
    viz_data = {}
    
    # 1. Daily Transaction Volume
    daily_volume = filtered_df.groupby('Date').size().reset_index(name='Count')
    fig1 = px.line(daily_volume, x='Date', y='Count',
                   title='Daily Transaction Volume',
                   labels={'Count': 'Number of Transactions', 'Date': 'Date'},
                   template='plotly_white')
    fig1.update_traces(line=dict(width=3))
    viz_data['daily_volume'] = fig1
    
    # 2. Top Products Distribution
    top_products = filtered_df['Product Name'].value_counts().head(10)
    fig2 = px.bar(x=top_products.values, y=top_products.index,
                  orientation='h',
                  title='Top 10 Products by Transaction Count',
                  labels={'x': 'Transaction Count', 'y': 'Product'},
                  template='plotly_white')
    fig2.update_traces(marker_color='#3B82F6')
    viz_data['top_products'] = fig2
    
    # 3. Hourly Distribution
    hourly_dist = filtered_df['Hour'].value_counts().sort_index()
    fig3 = px.line(x=hourly_dist.index, y=hourly_dist.values,
                   title='Hourly Transaction Distribution',
                   labels={'x': 'Hour of Day', 'y': 'Transaction Count'},
                   template='plotly_white')
    fig3.update_traces(line=dict(width=3, color='#10B981'))
    viz_data['hourly_dist'] = fig3
    
    # 4. Customer Segmentation
    if 'Entity Name' in filtered_df.columns:
        customer_types = filtered_df['Entity Name'].fillna('Unknown')
        customer_counts = customer_types.value_counts()
        fig4 = px.pie(values=customer_counts.values,
                      names=customer_counts.index,
                      title='Customer Type Distribution',
                      template='plotly_white')
        viz_data['customer_segmentation'] = fig4
    
    # 5. Weekday Analysis
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_data = filtered_df['Day'].value_counts().reindex(weekday_order)
    fig5 = px.bar(x=weekday_data.index, y=weekday_data.values,
                  title='Transaction Volume by Weekday',
                  labels={'x': 'Day', 'y': 'Transaction Count'},
                  template='plotly_white')
    fig5.update_traces(marker_color='#8B5CF6')
    viz_data['weekday_analysis'] = fig5
    
    return viz_data

def analyze_telemarketing_data(filtered_df):
    """Main analysis function"""
    # Define product categories
    p2p_products = ['Internal Wallet Transfer (P2P)', 'Internal Wallet Transfer', 'P2P Transfer']
    international_remittance = ['International Remittance', 'International Transfer']
    other_services = [
        'Deposit', 'Scan To Withdraw Agent', 'Scan To Withdraw Customer', 
        'Scan To Send', 'Ticket', 'Cash In', 'Cash Out'
    ]
    bill_payment_services = ['Bill Payment', 'Airtime Topup', 'Utility Payment']
    
    # Get unique customers
    unique_customers = filtered_df['User Identifier'].dropna().unique()
    
    # REPORT 1: International remittance customers not using P2P
    intl_mask = filtered_df['Product Name'].str.contains('|'.join(international_remittance), case=False, na=False)
    intl_customers = filtered_df[intl_mask]['User Identifier'].dropna().unique()
    
    p2p_mask = filtered_df['Product Name'].isin(p2p_products)
    p2p_customers = filtered_df[p2p_mask]['User Identifier'].dropna().unique()
    
    intl_not_p2p = [cust for cust in intl_customers if cust not in p2p_customers]
    
    # Create detailed DataFrame for international customers not using P2P
    intl_details = []
    for cust_id in intl_not_p2p[:1000]:  # Limit to 1000 for performance
        cust_records = filtered_df[filtered_df['User Identifier'] == cust_id]
        if not cust_records.empty:
            # Get customer name
            name_record = cust_records[cust_records['Full Name'].notna()]
            full_name = name_record.iloc[0]['Full Name'] if not name_record.empty else 'Name not available'
            
            # Get last transaction date
            last_date = cust_records['Created At'].max()
            last_date_str = last_date.strftime('%Y-%m-%d') if pd.notna(last_date) else 'Unknown'
            
            # Calculate metrics
            total_transactions = len(cust_records)
            intl_count = len(cust_records[intl_mask])
            
            intl_details.append({
                'User Identifier': cust_id,
                'Full Name': full_name,
                'Last Transaction Date': last_date_str,
                'Total Transactions': total_transactions,
                'International Transactions': intl_count,
                'Average Transaction Interval': np.nan,  # Could be calculated if needed
                'Customer Since': cust_records['Created At'].min().strftime('%Y-%m-%d') if pd.notna(cust_records['Created At'].min()) else 'Unknown'
            })
    
    report1_df = pd.DataFrame(intl_details) if intl_details else pd.DataFrame(
        columns=['User Identifier', 'Full Name', 'Last Transaction Date', 'Total Transactions', 
                'International Transactions', 'Average Transaction Interval', 'Customer Since']
    )
    
    # REPORT 2: Non-international customers using other services but not P2P
    other_mask = filtered_df['Product Name'].isin(other_services) | \
                filtered_df['Service Name'].isin(bill_payment_services)
    
    other_customers = filtered_df[other_mask]['User Identifier'].dropna().unique()
    
    domestic_other_not_p2p = [
        cust for cust in other_customers 
        if (cust not in intl_customers) and (cust not in p2p_customers)
    ]
    
    domestic_details = []
    for cust_id in domestic_other_not_p2p[:1000]:  # Limit to 1000 for performance
        cust_records = filtered_df[filtered_df['User Identifier'] == cust_id]
        if not cust_records.empty:
            # Get customer name
            name_record = cust_records[cust_records['Full Name'].notna()]
            full_name = name_record.iloc[0]['Full Name'] if not name_record.empty else 'Name not available'
            
            # Get last transaction date
            last_date = cust_records['Created At'].max()
            last_date_str = last_date.strftime('%Y-%m-%d') if pd.notna(last_date) else 'Unknown'
            
            # Get services used
            services_used = cust_records['Product Name'].dropna().unique()
            other_services_list = [str(s) for s in services_used if str(s) in other_services]
            
            domestic_details.append({
                'User Identifier': cust_id,
                'Full Name': full_name,
                'Last Transaction Date': last_date_str,
                'Total Transactions': len(cust_records),
                'Other Services Used': ', '.join(other_services_list[:5]),
                'Other Services Count': len(other_services_list),
                'Customer Since': cust_records['Created At'].min().strftime('%Y-%m-%d') if pd.notna(cust_records['Created At'].min()) else 'Unknown'
            })
    
    report2_df = pd.DataFrame(domestic_details) if domestic_details else pd.DataFrame(
        columns=['User Identifier', 'Full Name', 'Last Transaction Date', 'Total Transactions',
                'Other Services Used', 'Other Services Count', 'Customer Since']
    )
    
    # REPORT 3: Daily Impact Report Template
    if len(filtered_df) > 0:
        start_date = filtered_df['Created At'].min().date()
        end_date = filtered_df['Created At'].max().date()
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
    else:
        dates = pd.date_range(end=datetime.now().date(), periods=7, freq='D')
    
    impact_template = []
    for date in dates:
        impact_template.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Total Customers Called': 0,
            'Customers Reached': 0,
            'Customers Not Reached': 0,
            'P2P Adoption After Call': 0,
            'Conversion Rate (%)': 0.0,
            'Average Transaction Value': 0.0,
            'Customer Feedback': '',
            'Follow-up Required': '',
            'Notes': ''
        })
    
    impact_df = pd.DataFrame(impact_template)
    
    # Summary statistics
    summary_stats = {
        'total_transactions': len(filtered_df),
        'unique_customers': len(unique_customers),
        'intl_customers': len(intl_customers),
        'intl_not_p2p': len(intl_not_p2p),
        'p2p_customers': len(p2p_customers),
        'other_customers': len(other_customers),
        'domestic_not_p2p': len(domestic_other_not_p2p),
        'start_date': filtered_df['Created At'].min().strftime('%Y-%m-%d') if len(filtered_df) > 0 else 'N/A',
        'end_date': filtered_df['Created At'].max().strftime('%Y-%m-%d') if len(filtered_df) > 0 else 'N/A'
    }
    
    return {
        'intl_not_p2p': report1_df,
        'domestic_other_not_p2p': report2_df,
        'impact_template': impact_df,
        'summary_stats': summary_stats
    }

def create_excel_download(results):
    """Create Excel file with all reports"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: International customers not using P2P
        if not results['intl_not_p2p'].empty:
            results['intl_not_p2p'].to_excel(writer, sheet_name='International_Not_P2P', index=False)
        
        # Sheet 2: Domestic customers not using P2P
        if not results['domestic_other_not_p2p'].empty:
            results['domestic_other_not_p2p'].to_excel(writer, sheet_name='Domestic_Other_Not_P2P', index=False)
        
        # Sheet 3: Impact template
        results['impact_template'].to_excel(writer, sheet_name='Daily_Impact_Template', index=False)
        
        # Sheet 4: Summary
        summary_df = pd.DataFrame({
            'Metric': [
                'Report Period Start',
                'Report Period End',
                'Total Transactions',
                'Total Unique Customers',
                'International Customers',
                'International Customers Not Using P2P',
                'P2P Users',
                'Other Services Users',
                'Domestic Customers Not Using P2P'
            ],
            'Value': [
                results['summary_stats']['start_date'],
                results['summary_stats']['end_date'],
                results['summary_stats']['total_transactions'],
                results['summary_stats']['unique_customers'],
                results['summary_stats']['intl_customers'],
                results['summary_stats']['intl_not_p2p'],
                results['summary_stats']['p2p_customers'],
                results['summary_stats']['other_customers'],
                results['summary_stats']['domestic_not_p2p']
            ]
        })
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 5: Recommendations
        recommendations = [
            "1. PRIORITY: Target international remittance customers first",
            "2. Focus on customers with recent transaction activity",
            "3. Bundle P2P promotion with existing services",
            "4. Offer incentives for first-time P2P users",
            "5. Track conversion rates daily",
            "6. Segment customers by transaction frequency",
            "7. Schedule calls during peak transaction hours",
            f"8. {results['summary_stats']['intl_not_p2p']} international customers need P2P education",
            f"9. {results['summary_stats']['domestic_not_p2p']} domestic customers are potential cross-sell targets"
        ]
        rec_df = pd.DataFrame({'Recommendations': recommendations})
        rec_df.to_excel(writer, sheet_name='Recommendations', index=False)
    
    output.seek(0)
    return output

def main():
    """Main Streamlit app"""
    st.markdown('<h1 class="main-header">📊 Telemarketing Campaign Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Transaction Data",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your transaction file (CSV or Excel)"
        )
        
        if uploaded_file is not None:
            # Load data
            df = load_data(uploaded_file)
            
            if df is not None:
                # Date range filter
                st.subheader("📅 Date Range Filter")
                min_date = df['Created At'].min().date()
                max_date = df['Created At'].max().date()
                
                start_date = st.date_input(
                    "Start Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date
                )
                
                end_date = st.date_input(
                    "End Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date
                )
                
                # Product filter
                st.subheader("📦 Product Filter")
                unique_products = ['All'] + sorted(df['Product Name'].dropna().unique().tolist())[:50]
                product_filter = st.selectbox(
                    "Filter by Product",
                    options=unique_products,
                    index=0
                )
                
                # Customer type filter
                st.subheader("👥 Customer Type Filter")
                customer_type_filter = st.selectbox(
                    "Filter by Customer Type",
                    options=['All', 'Customer Only', 'Vendor/Agent Only'],
                    index=0
                )
                
                # Analyze button
                analyze_button = st.button(
                    "🚀 Analyze Data",
                    type="primary",
                    use_container_width=True
                )
                
                # Add sample data option
                with st.expander("📋 Sample Data Preview"):
                    st.dataframe(df.head(100), use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Rows", f"{len(df):,}")
                    with col2:
                        st.metric("Date Range", f"{min_date} to {max_date}")
        else:
            st.info("👈 Please upload a transaction file to begin analysis")
            df = None
            analyze_button = False
            start_date = end_date = product_filter = customer_type_filter = None
    
    # Main content area
    if uploaded_file is not None and df is not None and analyze_button:
        # Filter data
        filtered_df = filter_data(df, start_date, end_date, product_filter, customer_type_filter)
        
        if len(filtered_df) == 0:
            st.warning("⚠️ No data found for the selected filters. Please adjust your criteria.")
        else:
            # Display metrics
            st.markdown('<h2 class="sub-header">📈 Key Metrics</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Transactions", f"{len(filtered_df):,}")
            with col2:
                st.metric("Unique Customers", f"{filtered_df['User Identifier'].nunique():,}")
            with col3:
                st.metric("Date Range", f"{start_date} to {end_date}")
            with col4:
                st.metric("Products", filtered_df['Product Name'].nunique())
            
            # Create visualizations
            st.markdown('<h2 class="sub-header">📊 Data Visualizations</h2>', unsafe_allow_html=True)
            
            viz_data = create_visualizations(filtered_df, start_date, end_date)
            
            # Display charts in grid
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(viz_data['daily_volume'], use_container_width=True)
                st.plotly_chart(viz_data['hourly_dist'], use_container_width=True)
            with col2:
                st.plotly_chart(viz_data['top_products'], use_container_width=True)
                if 'customer_segmentation' in viz_data:
                    st.plotly_chart(viz_data['customer_segmentation'], use_container_width=True)
            
            st.plotly_chart(viz_data['weekday_analysis'], use_container_width=True)
            
            # Run analysis
            st.markdown('<h2 class="sub-header">🎯 Telemarketing Analysis Results</h2>', unsafe_allow_html=True)
            
            with st.spinner("Analyzing data for telemarketing targets..."):
                results = analyze_telemarketing_data(filtered_df)
            
            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "🎯 International Targets", 
                "🏠 Domestic Targets", 
                "📋 Impact Template", 
                "📊 Summary"
            ])
            
            with tab1:
                st.subheader("International Remittance Customers NOT Using P2P")
                st.dataframe(results['intl_not_p2p'], use_container_width=True)
                st.info(f"Found {len(results['intl_not_p2p'])} international customers not using P2P")
                
            with tab2:
                st.subheader("Domestic Customers Using Other Services But NOT P2P")
                st.dataframe(results['domestic_other_not_p2p'], use_container_width=True)
                st.info(f"Found {len(results['domestic_other_not_p2p'])} domestic customers not using P2P")
                
            with tab3:
                st.subheader("Daily Impact Report Template")
                st.dataframe(results['impact_template'], use_container_width=True)
                st.info("Use this template to track daily calling campaign results")
                
            with tab4:
                st.subheader("Campaign Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Key Statistics")
                    stats_df = pd.DataFrame({
                        'Metric': [
                            'International Customers',
                            'International Not Using P2P',
                            'Domestic Not Using P2P',
                            'P2P Users',
                            'Total Addressable Market'
                        ],
                        'Count': [
                            results['summary_stats']['intl_customers'],
                            results['summary_stats']['intl_not_p2p'],
                            results['summary_stats']['domestic_not_p2p'],
                            results['summary_stats']['p2p_customers'],
                            results['summary_stats']['intl_not_p2p'] + results['summary_stats']['domestic_not_p2p']
                        ]
                    })
                    st.dataframe(stats_df, use_container_width=True)
                
                with col2:
                    st.markdown("### 🎯 Campaign Focus Areas")
                    
                    # Create priority chart
                    priorities = pd.DataFrame({
                        'Segment': ['International Customers', 'Domestic Customers'],
                        'Count': [results['summary_stats']['intl_not_p2p'], 
                                 results['summary_stats']['domestic_not_p2p']],
                        'Priority': ['HIGH', 'MEDIUM']
                    })
                    
                    fig = px.bar(priorities, x='Segment', y='Count', color='Priority',
                                title='Target Segment Priority',
                                color_discrete_map={'HIGH': '#EF4444', 'MEDIUM': '#F59E0B'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                
                recommendations = [
                    f"**Priority 1**: Target {results['summary_stats']['intl_not_p2p']} international customers first",
                    f"**Priority 2**: Engage {results['summary_stats']['domestic_not_p2p']} domestic customers",
                    "**Strategy**: Bundle P2P with their existing services",
                    "**Timing**: Call during their typical transaction hours",
                    "**Incentive**: Offer fee waiver for first 3 P2P transactions"
                ]
                
                for rec in recommendations:
                    st.markdown(f"- {rec}")
            
            # Download section
            st.markdown('<h2 class="sub-header">📥 Download Reports</h2>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download Excel report
                excel_data = create_excel_download(results)
                st.download_button(
                    label="📊 Download Full Report (Excel)",
                    data=excel_data,
                    file_name=f"Telemarketing_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with col2:
                # Download CSV for international targets
                csv_intl = results['intl_not_p2p'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="🎯 Download International Targets (CSV)",
                    data=csv_intl,
                    file_name="international_targets.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col3:
                # Download CSV for domestic targets
                csv_domestic = results['domestic_other_not_p2p'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="🏠 Download Domestic Targets (CSV)",
                    data=csv_domestic,
                    file_name="domestic_targets.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Call-to-action
            st.markdown("---")
            st.markdown("### 🚀 Ready to Start Calling?")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Targets", 
                         f"{results['summary_stats']['intl_not_p2p'] + results['summary_stats']['domestic_not_p2p']:,}",
                         "customers to call")
            with col2:
                conversion_rate = (results['summary_stats']['intl_not_p2p'] / 
                                  results['summary_stats']['intl_customers'] * 100) if results['summary_stats']['intl_customers'] > 0 else 0
                st.metric("International Opportunity", 
                         f"{conversion_rate:.1f}%", 
                         "not using P2P")
            with col3:
                st.metric("Campaign Duration", 
                         f"{(end_date - start_date).days + 1} days",
                         "based on selected range")

if __name__ == "__main__":
    main()
